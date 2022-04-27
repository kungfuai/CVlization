import json
from typing import Union
import matplotlib.pyplot as plt
import cv2
import kornia as K

# import kornia.feature as KF
import numpy as np
import torch

# from kornia_moons.feature import *

# mkpts0 = correspondences["keypoints0"].cpu().numpy()
# mkpts1 = correspondences["keypoints1"].cpu().numpy()
# H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
# inliers = inliers > 0


# draw_LAF_matches(
#     KF.laf_from_center_scale_ori(
#         torch.from_numpy(mkpts0).view(1, -1, 2),
#         torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
#         torch.ones(mkpts0.shape[0]).view(1, -1, 1),
#     ),
#     KF.laf_from_center_scale_ori(
#         torch.from_numpy(mkpts1).view(1, -1, 2),
#         torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
#         torch.ones(mkpts1.shape[0]).view(1, -1, 1),
#     ),
#     torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
#     K.tensor_to_image(img1),
#     K.tensor_to_image(img2),
#     inliers,
#     draw_dict={
#         "inlier_color": (0.2, 1, 0.2),
#         "tentative_color": None,
#         "feature_color": (0.2, 0.5, 1),
#         "vertical": False,
#     },
# )


def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.0
    img = K.color.bgr_to_rgb(img)
    return img


class KorniaTransform:
    def __init__(self, config: Union[str, dict]):
        if isinstance(config, str):
            self.config = json.load(open(config))
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError(f"config must be a str or dict, not {type(config)}")

        self.transforms = []
        for t in self.config["transformers"]:
            self.transforms.append(self.get_tf(t))
        self.aug = torch.nn.Sequential(
            # K.enhance.Normalize(0.0, self._max_val),
            *self.transforms
        )

    def get_tf(self, transform_fn_and_kwargs: dict):
        class_name = transform_fn_and_kwargs["type"]
        kwargs = transform_fn_and_kwargs.get("kwargs", {})
        return getattr(K.augmentation, class_name)(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def transform(self, image, target=None):
        if target is None:
            return self.aug(image)
        return self.aug(image), target
