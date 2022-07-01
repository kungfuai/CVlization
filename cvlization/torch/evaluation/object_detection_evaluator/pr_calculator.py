from typing import Tuple, List, Optional
import torch
import numpy as np
from .confusion_matrix import ConfusionMatrix


class PRCalculator:
    def __init__(self, iou_detection_threshold: float = 0.5):
        self.iou_detection_threshold = iou_detection_threshold

    def _calculate_f1(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0
        else:
            return 2 * (precision * recall) / (precision + recall)

    def _concat_matched(self, matched_list: List[torch.Tensor]) -> torch.Tensor:
        """We concat the tensors together so that we can later count true pos, false pos, and false neg easier."""
        return torch.concat(matched_list, dim=1)

    def _concat_unmatched(
        self, unmatched_list: List[Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """We concat the tensors together so that we can later count true pos, false pos, and false neg easier."""
        unmatched = [item for item in unmatched_list if item != None]
        if len(unmatched) > 1:
            unmatched = torch.concat(unmatched, dim=0)
        elif len(unmatched) == 1:
            unmatched = unmatched[0]
        return unmatched

    def _analyze_matches(self, matched: torch.Tensor) -> Tuple[int, int, int]:
        """First discard detections lower than the score threshold. Then count the true pos, false pos, false neg.
        False negs determined in this method were matched to a target but had a IOU less than iou_detection_threshold
        """
        iou_row = matched[1, :]
        scores_row = matched[3, :]
        true_val_boolean = [iou_row >= self.iou_detection_threshold]
        false_val_boolean = [iou_row < self.iou_detection_threshold]
        true_positives = scores_row[true_val_boolean]
        false_positives = scores_row[false_val_boolean]
        false_negatives = scores_row[false_val_boolean]
        return true_positives, false_negatives, false_positives

    def _count_false_positives(
        self, unmatched: torch.Tensor, false_positives: int
    ) -> int:
        """We were also keeping a tensor of unmatched detections (detections not corresponding to a target as determined by
        MatchDeterminer._get_top2_dets or MatchDeterminer._get_top_det). We need to add these false positives to the total
        number of false positives"""
        return false_positives + len(unmatched)

    def _filter_by_score(self, arr, score):
        return arr[arr >= score]

    def _create_pr_array(self, true_positives, false_positives, false_negatives):
        possible_scores = torch.unique(
            torch.concat([true_positives, false_positives, false_negatives], dim=0)
        )
        possible_scores = torch.sort(possible_scores).values
        precision_recall_array = torch.zeros(size=(4, len(possible_scores)))
        for i, score in enumerate(possible_scores):
            true_positives = self._filter_by_score(arr=true_positives, score=score)
            false_positives = self._filter_by_score(arr=false_positives, score=score)
            false_negatives = self._filter_by_score(arr=false_negatives, score=score)

            confusion_matrix = ConfusionMatrix(
                true_positives=len(true_positives),
                false_negatives=len(false_negatives),
                false_positives=len(false_positives),
            )
            precision_recall_array[0, i] = confusion_matrix.precision
            precision_recall_array[1, i] = confusion_matrix.recall
            precision_recall_array[2, i] = self._calculate_f1(
                precision=confusion_matrix.precision, recall=confusion_matrix.recall
            )
            precision_recall_array[3, i] = score
        return precision_recall_array

    def __call__(
        self, matched_list: List[torch.Tensor], unmatched_list: List[torch.Tensor]
    ) -> np.ndarray:
        """
        This method takes in a matched list and an unmatched list and returns a [4xN] tensor
        containing the following:
            arr[0,i] = precision
            arr[1,i] = recall
            arr[2,i] = f1 score
            arr[3,i] = score_threshold (confidence) associated with the precision
            and recall values
        """
        matched = self._concat_matched(matched_list=matched_list)
        unmatched = self._concat_unmatched(unmatched_list=unmatched_list)
        true_positives, false_negatives, false_positives = self._analyze_matches(
            matched=matched
        )
        false_positives = self._concat_unmatched([false_positives, unmatched])
        precision_recall_array = self._create_pr_array(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )
        return precision_recall_array.numpy()