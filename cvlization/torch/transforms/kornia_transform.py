import json
from typing import Union
import kornia as K

# import kornia.feature as KF
import numpy as np
import torch

from ...utils import getattr_recursively


class KorniaTransform:
    def __init__(self, config: Union[str, dict]):
        if isinstance(config, str):
            self.config = json.load(open(config))
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError(f"config must be a str or dict, not {type(config)}")

        self.transforms = []
        for t in self.config["steps"]:
            self.transforms.append(self.get_tf(t))
        self.aug = torch.nn.Sequential(
            # K.enhance.Normalize(0.0, self._max_val),
            *self.transforms
        )

    def get_tf(self, transform_fn_and_kwargs: dict):
        class_name = transform_fn_and_kwargs["type"]
        kwargs = transform_fn_and_kwargs.get("kwargs", {})
        create_augmentation_step = getattr_recursively(K, class_name)
        return create_augmentation_step(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def transform(self, image, target=None):
        if target is None:
            return self.aug(image)
        return self.aug(image), target


def example_kornia_augmentation_config():
    return {
        "steps": [
            {"type": "augmentation.RandomHorizontalFlip", "kwargs": {"p": 0.5}},
        ],
        "data_keys": [
            "image",
            "bbox",
        ],  # Accepts “input”, “mask”, “bbox”, “bbox_xyxy”, “bbox_xywh”, “keypoints”.
    }
