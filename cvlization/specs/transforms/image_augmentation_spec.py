from abc import abstractmethod
from dataclasses import dataclass
import enum
from typing import Union, List
from ...specs import ModelInput, ModelTarget


class ImageAugmentationProvider(str, enum.Enum):
    IMGAUG = "imgaug"  # DEPRECATED: imgaug has NumPy 2.x compatibility issues. Use ALBUMENTATIONS instead.
    TORCHVISION = "torchvision"
    KORNIA = "kornia"
    ALBUMENTATIONS = "albumentations"
    TENSORFLOW = "tensorflow"


@dataclass
class ImageAugmentationSpec:
    provider: ImageAugmentationProvider = ImageAugmentationProvider.IMGAUG
    # Using weak typing here for flexibility. Different providers can use
    # very different sets of configs. However, if we can find regular patterns
    # of how augmentations are specified across providers, we can use a stronger
    # type than dict.
    # `config`: path to the json config file, or a `dict` with the configuration.
    #
    config: Union[str, dict] = None
