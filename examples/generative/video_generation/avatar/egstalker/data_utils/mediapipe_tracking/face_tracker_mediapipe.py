"""
MediaPipe-based face tracking for EGSTalker preprocessing.
Replaces BFM 2009 dependency with MediaPipe Face Mesh + Face Geometry.

Outputs:
- track_params.pt: {euler, trans, vertices, focal}
- Pose data in OpenGL/Blender convention for NeRF/3DGS
"""

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import mediapipe as mp
from scipy.spatial.transform import Rotation


def rotation_matrix_to_euler_xyz(R):
    """
    Convert rotation matrix to Euler angles (XYZ intrinsic, matching EGSTalker).

    EGSTalker uses: rot_x * rot_y * rot_z
    """
    r = Rotation.from_matrix(R)
    euler = r.as_euler('xyz', degrees=False)  # radians
    return euler


def opencv_to_opengl_transform(opencv_matrix):
    """
    Convert OpenCV convention (+Z forward) to OpenGL/Blender convention (+Z back).

    OpenCV: +X right, +Y down, +Z forward (camera looks +Z)
    OpenGL: +X right, +Y up, +Z back (camera looks -Z)

    Transformation: flip Y and Z axes
    """
    # Flip matrix: negate row 1 and row 2
    flip = np.diag([1, -1, -1, 1])
    return flip @ opencv_matrix @ flip


def estimate_focal_length(image_width, image_height):
    """
    Estimate focal length for perspective camera.
    Common approximation: f ≈ 1.2 * max(width, height)
    """
    return 1.2 * max(image_width, image_height)


def track_faces_mediapipe(ori_imgs_dir, output_dir, img_h=512, img_w=512):
    """
    Track faces using MediaPipe Face Mesh with temporal smoothing.

    Args:
        ori_imgs_dir: Directory containing extracted frames (*.jpg)
        output_dir: Base directory to save track_params.pt
        img_h: Image height
        img_w: Image width

    Returns:
        Path to saved track_params.pt
    """
    print(f'[INFO] ===== MediaPipe face tracking started =====')

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh

    # Get image paths
    import glob
    image_paths = sorted(glob.glob(os.path.join(ori_imgs_dir, '*.jpg')))
    num_frames = len(image_paths)

    if num_frames == 0:
        raise ValueError(f"No images found in {ori_imgs_dir}")

    print(f"[INFO] Processing {num_frames} frames...")

    # Storage for tracking results
    euler_angles_list = []
    translations_list = []
    vertices_list = []  # Store vertices for each frame
    canonical_vertices = None

    # Estimate focal length
    focal_length = estimate_focal_length(img_w, img_h)
    cx, cy = img_w / 2.0, img_h / 2.0

    # Previous frame's pose and vertices for temporal consistency
    prev_rvec = None
    prev_tvec = None
    prev_vertices = None

    # Smoothing parameters
    alpha_pose = 0.7  # Exponential smoothing for pose (0=no smoothing, 1=full smoothing)
    alpha_verts = 0.5  # Exponential smoothing for vertices (lighter to preserve expression)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        for idx, img_path in enumerate(tqdm(image_paths, desc="Tracking faces")):
            # Initialize frame vertices to None
            frame_vertices = None

            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"[WARN] Failed to read {img_path}, skipping")
                continue

            # Resize if needed
            if image.shape[0] != img_h or image.shape[1] != img_w:
                image = cv2.resize(image, (img_w, img_h))

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = face_mesh.process(image_rgb)

            if not results.multi_face_landmarks:
                # Use previous pose if available, else identity
                if prev_rvec is not None:
                    rvec, tvec = prev_rvec.copy(), prev_tvec.copy()
                else:
                    euler_angles_list.append(np.zeros(3))
                    translations_list.append(np.array([0.0, 0.0, -10.0]))
                    # Use canonical vertices if available, else zeros
                    if canonical_vertices is not None:
                        vertices_list.append(canonical_vertices.copy())
                    else:
                        vertices_list.append(np.zeros((468, 3)))
                    continue
            else:
                # Get face landmarks
                face_landmarks = results.multi_face_landmarks[0]

                # Extract 3D landmarks in normalized space [0, 1] for x,y and relative depth for z
                landmarks_3d = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])

                # Store canonical vertices from first frame (in normalized space)
                if canonical_vertices is None:
                    # Scale to metric space (assume face width ~15cm in canonical space)
                    face_scale = 15.0  # cm, typical face width
                    canonical_vertices = landmarks_3d.copy()
                    # Center and scale
                    canonical_vertices[:, :2] = (canonical_vertices[:, :2] - 0.5) * face_scale
                    canonical_vertices[:, 2] = canonical_vertices[:, 2] * face_scale * 0.3  # depth scale

                # Scale current frame's landmarks the same way and store them
                frame_vertices = landmarks_3d.copy()
                frame_vertices[:, :2] = (frame_vertices[:, :2] - 0.5) * face_scale
                frame_vertices[:, 2] = frame_vertices[:, 2] * face_scale * 0.3

                # Convert normalized coordinates to pixel coordinates
                points_2d = landmarks_3d[:, :2].copy()
                points_2d[:, 0] *= img_w
                points_2d[:, 1] *= img_h

                # Use canonical vertices as 3D model
                points_3d = canonical_vertices.copy()

                # Camera intrinsics matrix
                camera_matrix = np.array([
                    [focal_length, 0, cx],
                    [0, focal_length, cy],
                    [0, 0, 1]
                ], dtype=np.float32)

                # Solve PnP with previous frame as initial guess for stability
                dist_coeffs = np.zeros(4)
                if prev_rvec is not None:
                    success, rvec, tvec = cv2.solvePnP(
                        points_3d,
                        points_2d,
                        camera_matrix,
                        dist_coeffs,
                        rvec=prev_rvec,
                        tvec=prev_tvec,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                else:
                    success, rvec, tvec = cv2.solvePnP(
                        points_3d,
                        points_2d,
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                if not success:
                    # Use previous pose
                    if prev_rvec is not None:
                        rvec, tvec = prev_rvec.copy(), prev_tvec.copy()
                    else:
                        euler_angles_list.append(np.zeros(3))
                        translations_list.append(np.array([0.0, 0.0, -10.0]))
                        # Use canonical vertices if available
                        if canonical_vertices is not None:
                            vertices_list.append(canonical_vertices.copy())
                        else:
                            vertices_list.append(np.zeros((468, 3)))
                        continue

                # Apply temporal smoothing to pose
                if prev_rvec is not None and idx > 0:
                    rvec = alpha_pose * prev_rvec + (1 - alpha_pose) * rvec
                    tvec = alpha_pose * prev_tvec + (1 - alpha_pose) * tvec

                # Apply temporal smoothing to vertices
                if prev_vertices is not None and idx > 0:
                    frame_vertices = alpha_verts * prev_vertices + (1 - alpha_verts) * frame_vertices

                # Store for next frame
                prev_rvec = rvec.copy()
                prev_tvec = tvec.copy()
                if frame_vertices is not None:
                    prev_vertices = frame_vertices.copy()

            # Convert rotation vector to matrix
            R_opencv, _ = cv2.Rodrigues(rvec)

            # Build 4x4 transform matrix (face-to-camera in OpenCV convention)
            T_opencv = np.eye(4)
            T_opencv[:3, :3] = R_opencv
            T_opencv[:3, 3] = tvec.squeeze()

            # Convert OpenCV to OpenGL convention
            T_opengl = opencv_to_opengl_transform(T_opencv)

            # Extract rotation and translation
            R = T_opengl[:3, :3]
            t = T_opengl[:3, 3]

            # Convert rotation matrix to Euler angles (XYZ order)
            euler = rotation_matrix_to_euler_xyz(R)

            # Unwrap angles to prevent jumps at ±π
            if len(euler_angles_list) > 0:
                prev_euler = euler_angles_list[-1]
                for i in range(3):
                    while euler[i] - prev_euler[i] > np.pi:
                        euler[i] -= 2 * np.pi
                    while euler[i] - prev_euler[i] < -np.pi:
                        euler[i] += 2 * np.pi

            euler_angles_list.append(euler)
            translations_list.append(t)

            # Store vertices for this frame
            if frame_vertices is not None:
                vertices_list.append(frame_vertices)
            elif canonical_vertices is not None:
                # Use canonical if frame detection failed but pose succeeded
                vertices_list.append(canonical_vertices.copy())
            else:
                # Fallback: zero vertices
                vertices_list.append(np.zeros((468, 3)))

    # Convert to tensors
    euler_tensor = torch.tensor(np.array(euler_angles_list), dtype=torch.float32)
    trans_tensor = torch.tensor(np.array(translations_list), dtype=torch.float32)
    vertices_tensor = torch.tensor(np.array(vertices_list), dtype=torch.float32)  # [N, 468, 3]
    focal_tensor = torch.tensor([focal_length], dtype=torch.float32)

    # Save track_params.pt
    track_params = {
        'euler': euler_tensor,      # [N, 3] Euler angles XYZ
        'trans': trans_tensor,       # [N, 3] Translation vectors
        'vertices': vertices_tensor, # [N, 468, 3] Per-frame face mesh vertices
        'focal': focal_tensor        # [1] Focal length
    }

    output_path = os.path.join(output_dir, 'track_params.pt')
    torch.save(track_params, output_path)

    print(f'[INFO] Saved track_params.pt with:')
    print(f'  - {len(euler_angles_list)} frames')
    print(f'  - {canonical_vertices.shape[0]} vertices')
    print(f'  - focal length: {focal_length:.2f}')
    print(f'[INFO] ===== MediaPipe face tracking finished =====')

    return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MediaPipe face tracking for EGSTalker')
    parser.add_argument('--path', type=str, required=True, help='Path to ori_imgs directory')
    parser.add_argument('--img_h', type=int, default=512, help='Image height')
    parser.add_argument('--img_w', type=int, default=512, help='Image width')
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: parent of path)')

    args = parser.parse_args()

    output_dir = args.output if args.output else os.path.dirname(args.path)

    track_faces_mediapipe(args.path, output_dir, args.img_h, args.img_w)
