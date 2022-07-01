from cvlization.torch.evaluation.object_detection_evaluator.dict_to_tensor import DictToTensor
from tests.torch.object_detection_evaluator.multi_class_fixtures import (
    targets_multi_class,
    detections_multi_class,
)
import torch

EXPECTED_TARGETS = [
    torch.tensor(
        [
            [12.0, 51.0, 17.0, 56.0, 0.0],
            [42.0, 113.0, 47.0, 118.0, 1.0],
            [114.0, 52.0, 119.0, 57.0, 1.0],
        ]
    ),
    torch.tensor(
        [
            [15.0, 23.0, 20.0, 28.0, 0.0],
            [19.0, 85.0, 24.0, 90.0, 0.0],
            [63.0, 3.0, 68.0, 8.0, 1.0],
        ]
    ),
]

EXPECTED_DETECTIONS = [
    torch.tensor(
        [
            [1.2000e01, 5.1000e01, 1.7000e01, 5.6000e01, 0.0000e00, 1.0000e00],
            [1.3000e01, 5.1000e01, 1.8000e01, 5.6000e01, 0.0000e00, 7.0000e-01],
            [4.2000e01, 1.1300e02, 4.7000e01, 1.1800e02, 1.0000e00, 1.0000e-01],
            [1.1600e02, 5.2000e01, 1.2100e02, 5.7000e01, 1.0000e00, 9.0000e-01],
        ]
    ),
    torch.tensor(
        [
            [15.0000, 23.0000, 20.0000, 28.0000, 0.0000, 0.7000],
            [19.0000, 85.0000, 24.0000, 90.0000, 1.0000, 0.8000],
            [63.0000, 3.0000, 68.0000, 8.0000, 1.0000, 0.9000],
        ]
    ),
]


def test_call():
    dict_to_tensor = DictToTensor()
    targets, detections = dict_to_tensor(
        targets=targets_multi_class, detections=detections_multi_class
    )
    assert targets[0].sum() == EXPECTED_TARGETS[0].sum()
    assert targets[1].sum() == EXPECTED_TARGETS[1].sum()

    assert detections[0].sum() == EXPECTED_DETECTIONS[0].sum()
    assert detections[1].sum() == EXPECTED_DETECTIONS[1].sum()
