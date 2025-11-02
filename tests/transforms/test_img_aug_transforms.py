import numpy as np
import pytest
from pathlib import Path

from cvlization.transforms.albumentations_transform import AlbumentationsTransform
from cvlization.transforms.example_transform import ExampleTransform
from cvlization.specs import ModelInput, ModelTarget, DataColumnType

def random_image(height: int = 320, width: int = 480) -> np.ndarray:
    return np.random.randint(0, 256, size=(3, height, width), dtype=np.uint8)

def random_mask(height: int = 320, width: int = 480) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.int32)
    mask[50:150, 60:160] = 1
    return mask

def random_boxes() -> np.ndarray:
    # Single pascal VOC style bounding box (xmin, ymin, xmax, ymax)
    return np.array([[30, 40, 180, 200]], dtype=np.float32)

CONFIG_PATH = Path("tests/transforms/example_albumentations_config.json")

def test_albumentations_bbox_transform():
    transform = AlbumentationsTransform(
        cv_task="detection", config_file_or_dict=CONFIG_PATH
    )
    image = random_image()
    boxes = random_boxes()
    result = transform.transform_image_and_targets(image=image, bboxes=boxes)
    assert result["image"].shape[0] == 3
    assert result["bboxes"].shape == (1, 4)
    # ensure boxes remain within image bounds
    h, w = result["image"].shape[1:]
    assert np.all(result["bboxes"][:, 0] >= 0)
    assert np.all(result["bboxes"][:, 1] >= 0)
    assert np.all(result["bboxes"][:, 2] <= w)
    assert np.all(result["bboxes"][:, 3] <= h)

def test_albumentations_semseg_transform():
    transform = AlbumentationsTransform(
        cv_task="semseg", config_file_or_dict=CONFIG_PATH
    )
    image = random_image()
    mask = random_mask()
    result = transform.transform_image_and_targets(image=image, mask=mask)
    assert result["image"].shape[0] == 3
    assert result["mask"].shape == result["image"].shape[1:]

def test_example_transform_integration_detection():
    transform = AlbumentationsTransform(
        cv_task="detection", config_file_or_dict=CONFIG_PATH
    )
    model_inputs = [
        ModelInput(
            key="image", column_type=DataColumnType.IMAGE, raw_shape=[3, None, None]
        )
    ]
    model_targets = [
        ModelTarget(
            key="boxes",
            column_type=DataColumnType.BOUNDING_BOXES,
            raw_shape=[None, 4],
        ),
        ModelTarget(
            key="labels",
            column_type=DataColumnType.NUMERICAL,
            raw_shape=[None],
        ),
    ]
    example_transform = ExampleTransform(
        model_inputs=model_inputs,
        model_targets=model_targets,
        image_augmentation=transform,
    )
    image = random_image()
    boxes = random_boxes()
    labels = np.ones(1, dtype=np.float32)
    transformed = example_transform.transform_example(([image], [boxes, labels]))
    aug_inputs, aug_targets = transformed
    assert isinstance(aug_inputs, list)
    assert aug_inputs[0].shape[0] == 3
    assert isinstance(aug_targets, list)
    assert aug_targets[0].shape[-1] == 4
