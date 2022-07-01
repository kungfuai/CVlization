import pickle
import torch
import numpy as np

with open("tests/torch/object_detection_evaluator/real_targets.pickle", "rb") as handle:
    real_targets = pickle.load(handle)
with open("tests/torch/object_detection_evaluator/real_detections.pickle", "rb") as handle:
    real_detections = pickle.load(handle)


targets = [
    {
        "boxes": torch.tensor(
            [
                [12.0, 51.0, 17.0, 56.0],
                [42.0, 113.0, 47.0, 118.0],
                [114.0, 52.0, 119.0, 57.0],
            ]
        ),
        "labels": torch.tensor([0, 0, 0]),
    },
    {
        "boxes": torch.tensor(
            [
                [15.0, 23.0, 20.0, 28.0],
                [19.0, 85.0, 24.0, 90.0],
                [63.0, 3.0, 68.0, 8.0],
            ]
        ),
        "labels": torch.tensor([0, 0, 0]),
    },
]

detections2 = [
    {
        "boxes": torch.tensor(
            [
                [12.0, 51.0, 17.0, 56.0],
                [13.0, 51.0, 18.0, 56.0],  # iou < 1 (filtered by nms)
                [42.0, 113.0, 47.0, 118.0],
                [116.0, 52.0, 121.0, 57.0],  # iou < 0.5 = miss
                [0, 0, 5, 5],  # iou = 0
            ]
        ),
        "scores": torch.tensor([1.0, 0.7, 0.1, 0.9, 0.1]),
        "labels": torch.tensor([0, 0, 0, 0, 0]),
    },
    {
        "boxes": torch.tensor(
            [
                [15.0, 23.0, 20.0, 28.0],
                [19.0, 85.0, 24.0, 90.0],
                [63.0, 3.0, 68.0, 8.0],
            ]
        ),
        "scores": torch.tensor([0.7, 0.8, 0.9]),
        "labels": torch.tensor([0, 0, 0]),
    },
]

targets_with_empty = [
    {
        "boxes": torch.tensor(
            [
                [12.0, 51.0, 17.0, 56.0],
                [42.0, 113.0, 47.0, 118.0],
                [114.0, 52.0, 119.0, 57.0],
            ]
        ),
        "labels": torch.tensor([0, 0, 0]),
    },
    {
        "boxes": torch.tensor(
            [
                [15.0, 23.0, 20.0, 28.0],
                [19.0, 85.0, 24.0, 90.0],
                [63.0, 3.0, 68.0, 8.0],
            ]
        ),
        "labels": torch.tensor([0, 0, 0]),
    },
    {"boxes": torch.tensor([]), "labels": torch.tensor([])},
]

detections2_plus_one = [
    {
        "boxes": torch.tensor(
            [
                [12.0, 51.0, 17.0, 56.0],
                [13.0, 51.0, 18.0, 56.0],  # iou < 1 (filtered by nms)
                [42.0, 113.0, 47.0, 118.0],
                [116.0, 52.0, 121.0, 57.0],  # iou < 0.5 = miss
                [0, 0, 5, 5],  # iou = 0
            ]
        ),
        "scores": torch.tensor([1.0, 0.7, 0.1, 0.9, 0.1]),
        "labels": torch.tensor([0, 0, 0, 0, 0]),
    },
    {
        "boxes": torch.tensor(
            [
                [15.0, 23.0, 20.0, 28.0],
                [19.0, 85.0, 24.0, 90.0],
                [63.0, 3.0, 68.0, 8.0],
            ]
        ),
        "scores": torch.tensor([0.7, 0.8, 0.9]),
        "labels": torch.tensor([0, 0, 0]),
    },
    {
        "boxes": torch.tensor(
            [
                [15.0, 23.0, 20.0, 28.0],
                [19.0, 85.0, 24.0, 90.0],
                [63.0, 3.0, 68.0, 8.0],
            ]
        ),
        "scores": torch.tensor([0.7, 0.8, 0.9]),
        "labels": torch.tensor([0, 0, 0]),
    },
]

targets_plus_one = [
    {
        "boxes": torch.tensor(
            [
                [12.0, 51.0, 17.0, 56.0],
                [42.0, 113.0, 47.0, 118.0],
                [114.0, 52.0, 119.0, 57.0],
            ]
        ),
        "labels": torch.tensor([0, 0, 0]),
    },
    {
        "boxes": torch.tensor(
            [
                [15.0, 23.0, 20.0, 28.0],
                [19.0, 85.0, 24.0, 90.0],
                [63.0, 3.0, 68.0, 8.0],
            ]
        ),
        "labels": torch.tensor([0, 0, 0]),
    },
    {
        "boxes": torch.tensor(
            [
                [15.0, 23.0, 20.0, 28.0],
                [19.0, 85.0, 24.0, 90.0],
                [63.0, 3.0, 68.0, 8.0],
            ]
        ),
        "scores": torch.tensor([0.7, 0.8, 0.9]),
        "labels": torch.tensor([0, 0, 0]),
    },
]

detections2_with_empty = [
    {
        "boxes": torch.tensor(
            [
                [12.0, 51.0, 17.0, 56.0],
                [13.0, 51.0, 18.0, 56.0],  # iou < 1 (filtered by nms)
                [42.0, 113.0, 47.0, 118.0],
                [116.0, 52.0, 121.0, 57.0],  # iou < 0.5 = miss
                [0, 0, 5, 5],  # iou = 0
            ]
        ),
        "scores": torch.tensor([1.0, 0.7, 0.1, 0.9, 0.1]),
        "labels": torch.tensor([0, 0, 0, 0, 0]),
    },
    {
        "boxes": torch.tensor(
            [
                [15.0, 23.0, 20.0, 28.0],
                [19.0, 85.0, 24.0, 90.0],
                [63.0, 3.0, 68.0, 8.0],
            ]
        ),
        "scores": torch.tensor([0.7, 0.8, 0.9]),
        "labels": torch.tensor([0, 0, 0]),
    },
    {
        "boxes": torch.tensor([]),
        "scores": torch.tensor([]),
        "labels": torch.tensor([]),
    },
]

detections = [
    {
        "boxes": torch.tensor(
            [
                [12.0, 51.0, 17.0, 56.0],
                [13.0, 51.0, 18.0, 56.0],  # iou < 1 (filtered by nms)
                [42.0, 113.0, 47.0, 118.0],
                [116.0, 52.0, 121.0, 57.0],  # iou < 0.5 = miss
            ]
        ),
        "scores": torch.tensor([1.0, 0.7, 0.1, 0.9]),
        "labels": torch.tensor([0, 0, 0, 0]),
    },
    {
        "boxes": torch.tensor(
            [
                [15.0, 23.0, 20.0, 28.0],
                [19.0, 85.0, 24.0, 90.0],
                [63.0, 3.0, 68.0, 8.0],
            ]
        ),
        "scores": torch.tensor([0.7, 0.8, 0.9]),
        "labels": torch.tensor([0, 0, 0]),
    },
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

matched_list = [
    torch.tensor(
        [
            [0.0000, 1.0000, 2.0000],
            [1.0000, 1.0000, 0.4286],
            [0.0000, 1.0000, 2.0000],
            [1.0000, 0.1000, 0.9000],
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
    torch.empty(size=(4, 0)),
    torch.tensor(
        [
            [0.0000, 1.0000, 2.0000],
            [1.0000, 1.0000, 0.4286],
            [0.0000, 1.0000, 2.0000],
            [1.0000, 0.1000, 0.9000],
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
    torch.empty(size=(4, 0)),
]

unmatched_list = [torch.tensor([0.1000]), None, torch.tensor([0.1000]), None]


pr_array_list = [
    np.array(
        [
            [0.71428573, 0.8333333, 1.0],
            [0.8333333, 0.8333333, 1.0],
            [0.7692308, 0.8333333, 1.0],
            [0.1, 0.42857143, 1.0],
        ],
        dtype=np.float32,
    )
]
all_classes_pr_array_list = [
    np.array(
        [
            [0.71428573, 0.8333333, 1.0],
            [0.8333333, 0.8333333, 1.0],
            [0.7692308, 0.8333333, 1.0],
            [0.1, 0.42857143, 1.0],
        ],
        dtype=np.float32,
    )
]
