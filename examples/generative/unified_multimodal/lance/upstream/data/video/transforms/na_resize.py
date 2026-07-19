# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Literal
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize

from .area_resize import AreaResize
from .bucket_resize import BucketResize

def NaResize(
    resolution: int,
    mode: Literal["area", "square", "bucket"],
    downsample_only: bool,
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    **kwargs,
):
    if mode == "area":
        return AreaResize(
            max_area=resolution**2,
            downsample_only=downsample_only,
            interpolation=interpolation,
        )
    elif mode == "square":
        return Compose(
            [
                Resize(
                    size=resolution,
                    interpolation=interpolation,
                ),
                CenterCrop(resolution),
            ]
        )
    elif mode == "bucket":
        aspect_ratios = kwargs.get("aspect_ratios", ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"])
        stride = kwargs.get("stride", 16)
        return Compose(
            [
                BucketResize(
                    max_area=resolution**2,
                    interpolation=interpolation,
                    aspect_ratios=aspect_ratios,
                    stride=stride,
                )
            ]
        )
    raise ValueError(f"Unknown resize mode: {mode}")
