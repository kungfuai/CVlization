import torch
import numpy as np
from numpy import float64, ndarray
from typing import List, Optional


class Metrics:
    def __init__(self, score_thresholds: Optional[ndarray] = None) -> None:
        self.score_thresholds = score_thresholds
        self._num_images = 0
        self._f1_max = 0
        self._score_at_f1_max = 0
        self._map = 0

    @property
    def num_images(self):
        return self._num_images

    @property
    def f1_max(self) -> float64:
        """The F1 score is the F-score at the harmonic mean of precision and recall. F1 scores
        are calculated for various score (confidence) thresholds. The F1_max is the maximum of
        these F1 scores.

        If there are multiple classes, this value will be for all classes together. The reason for
        this is that the maximum F1 score for each class will occur at different score thresholds. It
        would be a fallacy to thus average f1_max across classes. So in order to get an overall F1_max,
        we consider all classes together.
        """
        return self._f1_max

    @property
    def mean_average_precision(self) -> float64:
        """The average of the average precision of each class, averaged across classes"""
        return self._map

    @property
    def score_at_f1_max(self):
        """The score threshold (confidence) at which the maximum F1 score occurs"""
        return self._score_at_f1_max

    def update_metrics(
        self,
        pr_array_list: List[ndarray],
        all_classes_precision_recall_array: ndarray,
        recall_thresholds: ndarray,
    ) -> None:
        """This could have also been named 'calculate_metrics'. It calculates metrics."""
        self._f1_max = np.max(all_classes_precision_recall_array[2, :])
        self._score_at_f1_max = self.score_thresholds[
            np.where(all_classes_precision_recall_array[2, :] == self._f1_max)[0][0]
        ]
        self._map = self._get_mean_average_pecision(
            pr_array_list=pr_array_list,
            recall_thresholds=recall_thresholds,
        )

    def _get_mean_average_pecision(
        self, pr_array_list: List[ndarray], recall_thresholds: ndarray
    ) -> float64:
        """The smaller the interval between recall thresholds, the more precise the calculation
        of average precision"""
        ap_list = []
        for pr_array in pr_array_list:
            precision = pr_array[0, :]
            recall = pr_array[1, :]

            average_precision = 0
            for t in recall_thresholds:
                p = np.max(precision[recall >= t]) if np.sum(recall >= t) != 0 else 0
                average_precision = average_precision + p / recall_thresholds.size
            ap_list.append(average_precision)
        mean_average_precision = sum(ap_list) / len(ap_list)
        return mean_average_precision

    def reset(self):
        self._f1_max = 0
        self._score_at_f1_max = 0
        self._map = 0
        self._num_images = 0
