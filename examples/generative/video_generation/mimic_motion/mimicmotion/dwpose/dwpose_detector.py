import os

import numpy as np
import torch

from .wholebody import Wholebody
from ..utils.hf_cache import resolve_asset_path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DWposeDetector:
    """
    A pose detect method for image-like data.

    Parameters:
        model_det: (str) serialized ONNX format model path, 
                    such as https://huggingface.co/yzd-v/DWPose/blob/main/yolox_l.onnx
        model_pose: (str) serialized ONNX format model path, 
                    such as https://huggingface.co/yzd-v/DWPose/blob/main/dw-ll_ucoco_384.onnx
        device: (str) 'cpu' or 'cuda:{device_id}'
    """
    def __init__(self, model_det, model_pose, device='cpu'):
        self.model_det = model_det
        self.model_pose = model_pose
        self.device = device
        self._resolved_args = None

    def _get_args(self):
        if self._resolved_args is None:
            resolved_det = resolve_asset_path(
                self.model_det, env_override="DWPOSE_MODEL_DET"
            )
            resolved_pose = resolve_asset_path(
                self.model_pose, env_override="DWPOSE_MODEL_POSE"
            )
            self._resolved_args = (resolved_det, resolved_pose, self.device)
        return self._resolved_args

    def release_memory(self):
        if hasattr(self, 'pose_estimation'):
            del self.pose_estimation
            import gc; gc.collect()

    def __call__(self, oriImg):
        if not hasattr(self, 'pose_estimation'):
            self.pose_estimation = Wholebody(*self._get_args())

        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, score = self.pose_estimation(oriImg)
            nums, _, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            subset = score[:, :18].copy()
            for i in range(len(subset)):
                for j in range(len(subset[i])):
                    if subset[i][j] > 0.3:
                        subset[i][j] = int(18 * i + j)
                    else:
                        subset[i][j] = -1

            # un_visible = subset < 0.3
            # candidate[un_visible] = -1

            # foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            faces_score = score[:, 24:92]
            hands_score = np.vstack([score[:, 92:113], score[:, 113:]])

            bodies = dict(candidate=body, subset=subset, score=score[:, :18])
            pose = dict(bodies=bodies, hands=hands, hands_score=hands_score, faces=faces, faces_score=faces_score)

            return pose

dwpose_detector = DWposeDetector(
    model_det="hf://yzd-v/DWPose/yolox_l.onnx",
    model_pose="hf://yzd-v/DWPose/dw-ll_ucoco_384.onnx",
    device=device)
