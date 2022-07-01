import torch
from cvlization.torch.evaluation.object_detection_evaluator.pr_calculator import PRCalculator
from tests.torch.object_detection_evaluator.fixtures import matched_list, unmatched_list


def test_call():
    pr_calculator = PRCalculator(iou_detection_threshold=0.5)
    precision_recall_array = pr_calculator(
        matched_list=matched_list, unmatched_list=unmatched_list
    )
    assert torch.allclose(
        torch.from_numpy(precision_recall_array),
        torch.tensor(
            [
                [0.71428573, 0.8, 0.75, 0.6666667, 1],
                [0.8333333, 0.8, 0.75, 0.6666667, 1],
                [0.7692308, 0.8, 0.75, 0.6666667, 1],
                [0.1, 0.7, 0.8, 0.9, 1],
            ]
        ),
        rtol=1e-4,
    )


if __name__ == "__main__":
    test_call()
