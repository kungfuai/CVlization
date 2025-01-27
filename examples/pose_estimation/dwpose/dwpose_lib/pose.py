from tqdm import tqdm
import decord
import numpy as np
import torch

from .util import draw_pose
from .dwpose_detector import DWposeDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
dwprocessor = DWposeDetector(
    model_det="/root/.cache/models/DWPose/yolox_l.onnx",
    model_pose="/root/.cache/models/DWPose/dw-ll_ucoco_384.onnx",
    device=device)


def get_video_pose(
        video_path: str, 
        ref_image: np.ndarray=None, 
        sample_stride: int=1,
        draw: bool=False):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): video pose path
        ref_image (np.ndarray): reference image 
        sample_stride (int, optional): Defaults to 1.

    Returns:
        np.ndarray: sequence of video pose
    """
    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))

    frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    # frames = frames[:100]  # Uncomment to limit the number of frames. For debugging.
    if ref_image is None:
        # print("no reference image provided, using first frame as reference")
        ref_image = frames[0]

    # select ref-keypoint from reference pose for pose rescale
    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]
    # print("ref_body:", ref_body.shape)

    height, width, _ = ref_image.shape

    detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]
    dwprocessor.release_memory()

    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 18])[:,
                      ref_keypoint_id]
    # compute linear-rescale params
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_ref_pose = {
        "body": [],
        "face": [],
        "hand": []
    }
    output_pose_imgs = []
    # pose rescale 
    for detected_pose in tqdm(detected_poses, desc="drawing poses to a canvas"):
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b
        if draw:
            im = draw_pose(detected_pose, height, width)
            output_pose_imgs.append(np.array(im))
        output_ref_pose["body"].append(np.expand_dims(detected_pose["bodies"]["candidate"], axis=0))
        output_ref_pose["face"].append(detected_pose["faces"])
        output_ref_pose["hand"].append(detected_pose["hands"])
    
    # concat all pose arrays
    for key in output_ref_pose.keys():
        output_ref_pose[key] = np.concatenate(output_ref_pose[key], axis=0)
    
    # Add height and width to the pose
    output_ref_pose["height"] = height
    output_ref_pose["width"] = width
    return output_ref_pose, np.stack(output_pose_imgs) if draw else None


def get_image_pose(ref_image):
    """process image pose

    Args:
        ref_image (np.ndarray): reference image pixel value

    Returns:
        np.ndarray: pose visual image in RGB-mode
    """
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    pose_img = draw_pose(ref_pose, height, width)
    return ref_pose, np.array(pose_img)