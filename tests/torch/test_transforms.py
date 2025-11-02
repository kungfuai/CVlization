import numpy as np
import pytest
import torch
from cvlization.torch.transforms.kornia_transform import KorniaTransform
from cvlization.transforms.image_augmentation_builder import (
    ImageAugmentationBuilder,
    ImageAugmentationSpec,
    ImageAugmentationProvider,
)


def test_kornia_transform_creation_and_image_transform():
    config = {
        "steps": [
            {"type": "augmentation.RandomHorizontalFlip", "kwargs": {"p": 0.5}},
        ],
        "data_keys": ["image"],
    }
    kornia_transform = KorniaTransform(config)
    assert callable(kornia_transform)
    img = np.zeros((100, 200, 3))
    with pytest.raises(TypeError) as excinfo:
        kornia_transform(img)
    assert "Expected input of torch.Tensor" in str(excinfo.value)
    img = torch.tensor(np.zeros((100, 200, 3)))
    img_aug = kornia_transform(img)
    assert img_aug.shape == (1,) + img.shape

    img = torch.tensor(np.zeros((1, 100, 200, 3)))
    img_aug = kornia_transform(img)
    assert img_aug.shape == img.shape

    img = torch.tensor(np.zeros((2, 100, 200, 3)))
    img_aug = kornia_transform(img)
    assert img_aug.shape == img.shape


def test_kornia_bbox_transform():
    config = {
        "steps": [
            {"type": "augmentation.RandomHorizontalFlip", "kwargs": {"p": 0.5}},
            {"type": "augmentation.RandomAffine", "kwargs": {"degrees": 360, "p": 1}},
        ],
        "data_keys": ["image", "bbox"],
    }
    kornia_transform = KorniaTransform(config)
    img = torch.tensor(np.zeros((100, 200, 3)))
    bboxes = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45]])
    img_aug, bboxes_aug = kornia_transform(img, bboxes)
    assert img_aug.shape == (1,) + img.shape
    assert bboxes_aug.shape == bboxes_aug.shape  # no expand_dims happening!


def test_augmentation_builder_can_build_kornia():
    config = {
        "steps": [
            {"type": "augmentation.RandomHorizontalFlip", "kwargs": {"p": 0.5}},
            {"type": "augmentation.RandomAffine", "kwargs": {"degrees": 360, "p": 1}},
        ],
        "data_keys": ["image", "bbox"],
    }
    kornia_transform = ImageAugmentationBuilder(
        ImageAugmentationSpec(provider=ImageAugmentationProvider.KORNIA, config=config)
    ).run()
    img = torch.tensor(np.zeros((100, 200, 3)))
    bboxes = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45]])
    img_aug, bboxes_aug = kornia_transform(img, bboxes)
    assert img_aug.shape == (1,) + img.shape
    assert bboxes_aug.shape == bboxes_aug.shape  # no expand_dims happening!
