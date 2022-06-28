import pickle
import torch
from .dict_to_tensor import DictToTensor


with open("tests/torch/object_detection_evaluator/real_targets.pickle", "rb") as handle:
    real_targets = pickle.load(handle)
    real_targets = DictToTensor()(targets=real_targets)

with open("tests/torch/object_detection_evaluator/real_detections.pickle", "rb") as handle:
    real_detections = pickle.load(handle)
    real_detections = DictToTensor()(detections=real_detections)

targets = [
    torch.tensor(
        [
            [12.0, 51.0, 17.0, 56.0, 0],
            [42.0, 113.0, 47.0, 118.0, 0],
            [114.0, 52.0, 119.0, 57.0, 0],
        ]
    ),
    torch.tensor(
        [
            [15.0, 23.0, 20.0, 28.0, 0],
            [19.0, 85.0, 24.0, 90.0, 0],
            [63.0, 3.0, 68.0, 8.0, 0],
        ]
    ),
]

detections2 = [
    torch.tensor(
        [
            [12.0, 51.0, 17.0, 56.0, 0, 1.0],
            [13.0, 51.0, 18.0, 56.0, 0, 0.7],  # iou < 1 (filtered by nms)
            [42.0, 113.0, 47.0, 118.0, 0, 0.1],
            [116.0, 52.0, 121.0, 57.0, 0, 0.9],  # iou < 0.5 = miss
            [0, 0, 5, 50, 0, 0.1],  # iou = 0
        ]
    ),
    torch.tensor(
        [
            [15.0, 23.0, 20.0, 28.0, 0, 0.7],
            [19.0, 85.0, 24.0, 90.0, 0, 0.8],
            [63.0, 3.0, 68.0, 8.0, 0, 0.9],
        ]
    ),
]

targets_with_empty = [
    torch.tensor(
        [
            [12.0, 51.0, 17.0, 56.0, 0],
            [42.0, 113.0, 47.0, 118.0, 0],
            [114.0, 52.0, 119.0, 57.0, 0],
        ]
    ),
    torch.tensor(
        [
            [15.0, 23.0, 20.0, 28.0, 0],
            [19.0, 85.0, 24.0, 90.0, 0],
            [63.0, 3.0, 68.0, 8.0, 0],
        ]
    ),
    None,
]

detections2_plus_one = [
    torch.tensor(
        [
            [12.0, 51.0, 17.0, 56.0, 0, 1.0],
            [13.0, 51.0, 18.0, 56.0, 0, 0.7],  # iou < 1 (filtered by nms)
            [42.0, 113.0, 47.0, 118.0, 0, 0.1],
            [116.0, 52.0, 121.0, 57.0, 0, 0.9],  # iou < 0.5 = miss
            [0, 0, 5, 5, 0, 0.1],  # iou = 0
        ]
    ),
    torch.tensor(
        [
            [15.0, 23.0, 20.0, 28.0, 0, 0.7],
            [19.0, 85.0, 24.0, 90.0, 0, 0.8],
            [63.0, 3.0, 68.0, 8.0, 0, 0.9],
        ]
    ),
    torch.tensor(
        [
            [15.0, 23.0, 20.0, 28.0, 0, 0.7],
            [19.0, 85.0, 24.0, 90.0, 0, 0.8],
            [63.0, 3.0, 68.0, 8.0, 0, 0.9],
        ]
    ),
]

targets_plus_one = [
    torch.tensor(
        [
            [12.0, 51.0, 17.0, 56.0, 0],
            [42.0, 113.0, 47.0, 118.0, 0],
            [114.0, 52.0, 119.0, 57.0, 0],
        ]
    ),
    torch.tensor(
        [
            [15.0, 23.0, 20.0, 28.0, 0],
            [19.0, 85.0, 24.0, 90.0, 0],
            [63.0, 3.0, 68.0, 8.0, 0],
        ]
    ),
    torch.tensor(
        [
            [15.0, 23.0, 20.0, 28.0, 0],
            [19.0, 85.0, 24.0, 90.0, 0],
            [63.0, 3.0, 68.0, 8.0, 0],
        ]
    ),
]

detections2_with_empty = [
    torch.tensor(
        [
            [12.0, 51.0, 17.0, 56.0, 0, 1.0],
            [13.0, 51.0, 18.0, 56.0, 0, 0.7],  # iou < 1 (filtered by nms)
            [42.0, 113.0, 47.0, 118.0, 0, 0.1],
            [116.0, 52.0, 121.0, 57.0, 0, 0.9],  # iou < 0.5 = miss
            [0, 0, 5, 5, 0, 0.1],  # iou = 0
        ]
    ),
    torch.tensor(
        [
            [15.0, 23.0, 20.0, 28.0, 0, 0.7],
            [19.0, 85.0, 24.0, 90.0, 0, 0.8],
            [63.0, 3.0, 68.0, 8.0, 0, 0.9],
        ]
    ),
    None,
]

detections = [
    torch.tensor(
        [
            [12.0, 51.0, 17.0, 56.0, 0, 1.0],
            [13.0, 51.0, 18.0, 56.0, 0, 0.7],  # iou < 1 (filtered by nms)
            [42.0, 113.0, 47.0, 118.0, 0, 0.1],
            [116.0, 52.0, 121.0, 57.0, 0, 0.9],  # iou < 0.5 = miss
        ]
    ),
    torch.tensor(
        [
            [15.0, 23.0, 20.0, 28.0, 0, 0.7],
            [19.0, 85.0, 24.0, 90.0, 0, 0.8],
            [63.0, 3.0, 68.0, 8.0, 0, 0.9],
        ]
    ),
]


"""
Returns 4 TP, 1 FN, and 1 FP
"""
matched_output = [
    torch.tensor(
        [
            [0.0000, 1.0000, 2.0000],
            [1.0000, 1.0000, 0.4286],  # False negative when iou_threshold = 0.5
            [0.0000, 1.0000, 2.0000],
            [1.0000, 0.1000, 0.9000],  # False negative when score_threshold > 0.1
        ]
    ),
    torch.tensor(
        [
            [0.0000, 1.0000, 2.0000],
            [1.0000, 1.0000, 1.0000],
            [0.0000, 1.0000, 2.0000],
            [0.7000, 0.8000, 0.9000],
        ]
    ),
]

unmatched_output = [None, None]

unmatched_output2 = [torch.tensor([0.1000]), None]
