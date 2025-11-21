"""
Novel View Rendering for FastAvatar
====================================
Render novel views from generated Gaussian splats.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import json

from gsplat.rendering import rasterization
from utils import load_ply_to_splats


def create_camera_pose(elevation_deg, azimuth_deg, distance=2.5):
    """
    Create camera pose from spherical coordinates.

    Args:
        elevation_deg: Elevation angle in degrees (up/down)
        azimuth_deg: Azimuth angle in degrees (left/right rotation)
        distance: Distance from origin

    Returns:
        4x4 camera-to-world matrix
    """
    elevation = np.radians(elevation_deg)
    azimuth = np.radians(azimuth_deg)

    # Camera position in spherical coordinates
    x = distance * np.cos(elevation) * np.sin(azimuth)
    y = distance * np.sin(elevation)
    z = distance * np.cos(elevation) * np.cos(azimuth)

    camera_pos = np.array([x, y, z])

    # Look at origin
    forward = -camera_pos / np.linalg.norm(camera_pos)

    # Up vector (flipped Y to match Gaussian splat coordinate system)
    world_up = np.array([0.0, -1.0, 0.0])
    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    # Build camera-to-world matrix
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_pos

    # Convert to world-to-camera (view matrix)
    w2c = np.linalg.inv(c2w)

    return torch.from_numpy(w2c).float()  # Return full 4x4 matrix


def get_projection_matrix(width=550, height=802, fx=2049.778, fy=2049.768, cx=None, cy=None):
    """
    Create perspective projection matrix.

    Args:
        width: Image width
        height: Image height
        fx: Focal length in x (pixels)
        fy: Focal length in y (pixels)
        cx: Principal point x (default: width/2)
        cy: Principal point y (default: height/2)

    Returns:
        3x3 intrinsics matrix
    """
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    # Build 3x3 intrinsics matrix
    K = torch.tensor([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=torch.float32)

    return K


def render_view(splats, viewmat, K, width=550, height=802, device='cuda'):
    """
    Render a single view using gsplat.

    Args:
        splats: Dictionary of Gaussian splat parameters
        viewmat: 4x4 view matrix (world-to-camera)
        K: 3x3 intrinsics matrix
        width: Image width
        height: Image height
        device: Device to render on

    Returns:
        Rendered RGB image as numpy array
    """
    # Move everything to device
    viewmat = viewmat.to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4] - batch, num_cameras, 4, 4
    K = K.to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3] - batch, num_cameras, intrinsics

    means = splats['means'].to(device).unsqueeze(0)  # [1, N, 3]
    quats = splats['quats'].to(device).unsqueeze(0)  # [1, N, 4]
    scales = torch.exp(splats['scales']).to(device).unsqueeze(0)  # [1, N, 3]
    opacities = torch.sigmoid(splats['opacities']).to(device).unsqueeze(0)  # [1, N, 1]

    # Concatenate full spherical harmonics coefficients (sh0 + shN)
    # gsplat will handle SH-to-RGB conversion based on view direction
    sh0 = splats['sh0'].to(device)  # [N, 1, 3]
    shN = splats['shN'].to(device)  # [N, 15, 3]
    colors = torch.cat([sh0, shN], 1).unsqueeze(0)  # [1, N, 16, 3]

    # Render using gsplat with spherical harmonics
    with torch.no_grad():
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            packed=False,
            sh_degree=3,  # Use degree-3 spherical harmonics
            render_mode='RGB',
        )

    # Convert to numpy and format as image
    # Remove batch and camera dimensions [1, 1, H, W, 3] -> [H, W, 3]
    img = render_colors[0, 0].cpu().numpy()  # [H, W, 3]
    img = (img * 255).clip(0, 255).astype(np.uint8)

    return img


def main():
    parser = argparse.ArgumentParser(description='Render novel views from Gaussian splats')

    parser.add_argument('--ply_path', type=str, required=True,
                       help='Path to splats.ply file')
    parser.add_argument('--output_dir', type=str, default='novel_views',
                       help='Directory to save rendered views')
    parser.add_argument('--width', type=int, default=550,
                       help='Render width (default: 550 to match training data)')
    parser.add_argument('--height', type=int, default=802,
                       help='Render height (default: 802 to match training data)')
    parser.add_argument('--fx', type=float, default=2049.778,
                       help='Focal length x in pixels (default: 2049.778 from training data)')
    parser.add_argument('--fy', type=float, default=2049.768,
                       help='Focal length y in pixels (default: 2049.768 from training data)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--num_views', type=int, default=8,
                       help='Number of azimuth views to render')
    parser.add_argument('--elevation', type=float, default=0.0,
                       help='Elevation angle in degrees')
    parser.add_argument('--distance', type=float, default=1.0,
                       help='Camera distance from origin (default: 1.0)')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading splats from: {args.ply_path}")
    splats = load_ply_to_splats(args.ply_path)

    print(f"Loaded {splats['means'].shape[0]} Gaussians")

    # Get intrinsics with proper focal lengths matching training data
    K = get_projection_matrix(
        width=args.width,
        height=args.height,
        fx=args.fx,
        fy=args.fy
    )

    # Render views at different azimuth angles
    azimuth_angles = np.linspace(0, 360, args.num_views, endpoint=False)

    views_info = []

    for i, azimuth in enumerate(azimuth_angles):
        print(f"Rendering view {i+1}/{args.num_views}: azimuth={azimuth:.1f}°, elevation={args.elevation:.1f}°")

        # Create camera pose
        viewmat = create_camera_pose(
            elevation_deg=args.elevation,
            azimuth_deg=azimuth,
            distance=args.distance
        )

        # Render
        img = render_view(
            splats, viewmat, K,
            width=args.width,
            height=args.height,
            device=device
        )

        # Save image
        img_path = output_path / f'view_{i:03d}_az{azimuth:.0f}_el{args.elevation:.0f}.png'
        Image.fromarray(img).save(img_path)

        views_info.append({
            'index': i,
            'azimuth': float(azimuth),
            'elevation': float(args.elevation),
            'path': str(img_path)
        })

        print(f"  Saved: {img_path}")

    # Save metadata
    metadata = {
        'ply_path': args.ply_path,
        'num_gaussians': int(splats['means'].shape[0]),
        'render_width': args.width,
        'render_height': args.height,
        'num_views': args.num_views,
        'views': views_info
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Rendered {args.num_views} novel views")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
