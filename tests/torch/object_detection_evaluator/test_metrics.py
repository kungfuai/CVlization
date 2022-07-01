import numpy as np
from cvlization.torch.evaluation.object_detection_evaluator.metrics import Metrics
from tests.torch.object_detection_evaluator.fixtures import pr_array_list, all_classes_pr_array_list


def test_get_mean_average_precision_with_internet_example():
    """
    Using test example from: https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2
    """
    metrics = Metrics()

    precision = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    recall = np.arange(0.0, 1.1, 0.1)
    test_array = np.stack([precision, recall])

    mean_average_precision = metrics._get_mean_average_pecision(
        pr_array_list=[test_array]
    )
    assert round(mean_average_precision, 3) == 0.55


def test_get_mean_average_precision_with_real_example():
    metrics = Metrics()
    mean_average_precision = metrics._get_mean_average_pecision(pr_array_list)
    assert round(mean_average_precision, 3) == 0.748
    all_classes_mean_average_precision = metrics._get_mean_average_pecision(
        all_classes_pr_array_list
    )
    assert round(all_classes_mean_average_precision, 3) == 0.748


if __name__ == "__main__":
    test_get_mean_average_precision_with_internet_example()
