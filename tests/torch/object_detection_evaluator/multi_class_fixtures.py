import torch

targets_with_empty_cls0 = [
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

detections2_plus_one_cls0 = [
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

targets_plus_one_cls1 = [
    {
        "boxes": torch.tensor(
            [
                [12.0, 51.0, 17.0, 56.0],
                [42.0, 113.0, 47.0, 118.0],
                [114.0, 52.0, 119.0, 57.0],
            ]
        ),
        "labels": torch.tensor([1, 1, 1]),
    },
    {
        "boxes": torch.tensor(
            [
                [15.0, 23.0, 20.0, 28.0],
                [19.0, 85.0, 24.0, 90.0],
                [63.0, 3.0, 68.0, 8.0],
            ]
        ),
        "labels": torch.tensor([1, 1, 1]),
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
        "labels": torch.tensor([1, 1, 1]),
    },
]

detections2_with_empty_cls1 = [
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
        "labels": torch.tensor([1, 1, 1, 1, 1]),
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
        "labels": torch.tensor([1, 1, 1]),
    },
    {
        "boxes": torch.tensor([]),
        "scores": torch.tensor([]),
        "labels": torch.tensor([]),
    },
]

targets_multi_class = [
    {
        "boxes": torch.tensor(
            [
                [12.0, 51.0, 17.0, 56.0],
                [42.0, 113.0, 47.0, 118.0],
                [114.0, 52.0, 119.0, 57.0],
            ]
        ),
        "labels": torch.tensor([0, 1, 1]),
    },
    {
        "boxes": torch.tensor(
            [
                [15.0, 23.0, 20.0, 28.0],
                [19.0, 85.0, 24.0, 90.0],
                [63.0, 3.0, 68.0, 8.0],
            ]
        ),
        "labels": torch.tensor([0, 0, 1]),
    },
]


detections_multi_class = [  # class 0 =  1 TP, 0 FN, 0 FP  class 1 = 1 TP, 1 FN, 1 FP
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
        "labels": torch.tensor([0, 0, 1, 1]),
    },
    {
        "boxes": torch.tensor(  # class 0 =
            [
                [15.0, 23.0, 20.0, 28.0],
                [19.0, 85.0, 24.0, 90.0],
                [63.0, 3.0, 68.0, 8.0],
            ]
        ),
        "scores": torch.tensor([0.7, 0.8, 0.9]),
        "labels": torch.tensor([0, 1, 1]),
    },
]
