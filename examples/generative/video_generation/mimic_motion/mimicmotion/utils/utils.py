import logging
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

def save_to_mp4(frames, save_path, fps=7):
    # Accept torch tensor (frames, channels, height, width) in [0,255]
    if isinstance(frames, torch.Tensor):
        frames_np = frames.detach().cpu().permute(0, 2, 3, 1).numpy()
    else:
        frames_np = np.asarray(frames)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_np.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

    for frame in frames_np:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
