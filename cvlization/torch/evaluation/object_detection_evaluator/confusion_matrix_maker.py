from logging import exception
import torch
from src.evaluation.confusion_matrix import ConfusionMatrix
from numpy import float64
from typing import List, Optional, Tuple


class ConfusionMatrixMaker:
    """
    This class creates a confusion matrix from from lists of matched and unmatched detections

    __init__ Args:
        iou_detection_threshold (float): Minimum intersection-over-union for a detection to count as
                                        a true positive
        score_threshold (float): The threshold for score (confidence). Detections with scores lower
                                than score_threshold are discarded

    __call__ Args:
        matched (list[Tensor]): list of (N x 4) Tensors. Len(matched) = len(targets) = len(images).
            Tensor contains:
                [[Ground Truth Box Index],
                [IOU Between Ground Truth and Detection],
                [Detection Box Index],
                [Score for Detecton]]

        unmatched (list[Tensor]): list of (1 x N) Tensors.
            Tensor contains:
            [[score of unmatched detection 0], [score of unmatched detection 1], [score of unmatched detection 2], [etc]]

    __call__ Returns:
        confusion_matrix (ConfusionMatrix): Object containing num true_positives, false_negatives, false_positives
        *Also sets the class attribute self.confusion_matrix = confusion_matrix
    """

    def __init__(
        self, iou_detection_threshold: float = 0.5, score_threshold: float64 = 0.2
    ) -> None:
        self.iou_detection_threshold = iou_detection_threshold
        self.score_threshold = score_threshold
        self.confusion_matrix = ConfusionMatrix()

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
        filtered_matched = self._filter_by_score_threshold(matched=matched)
        iou_row = filtered_matched[1, :]
        true_positives = torch.numel(iou_row[iou_row > self.iou_detection_threshold])
        false_positives = filtered_matched.shape[1] - true_positives
        false_negatives = matched.shape[1] - true_positives

        return true_positives, false_negatives, false_positives

    def _filter_by_score_threshold(self, matched: torch.Tensor) -> torch.Tensor:
        filtered = torch.gt(matched[3:], self.score_threshold)
        filtered = torch.cat([filtered, filtered, filtered, filtered], dim=0)
        filtered_matched = matched[filtered]
        return torch.reshape(filtered_matched, (4, int(len(filtered_matched) / 4)))

    def _count_false_positives(
        self, unmatched: torch.Tensor, false_positives: int
    ) -> int:
        """We were also keeping a tensor of unmatched detections (detections not corresponding to a target as determined by
        MatchDeterminer._get_top2_dets or MatchDeterminer._get_top_det). We need to add these false positives to the total
        number of false positives"""
        return false_positives + len(unmatched)

    def __call__(
        self,
        matched_list: List[torch.Tensor],
        unmatched_list: List[Optional[torch.Tensor]],
    ) -> ConfusionMatrix:
        matched = self._concat_matched(matched_list=matched_list)
        unmatched = self._concat_unmatched(unmatched_list=unmatched_list)

        # Remove unmatched detections that are less than score_threshold
        if len(unmatched) > 0:
            unmatched = unmatched[unmatched > self.score_threshold]
        # Analyze the matched tensor to determine number of true pos, false pos, false neg
        true_positives, false_negatives, false_positives = self._analyze_matches(
            matched=matched
        )

        false_positives = self._count_false_positives(
            unmatched=unmatched, false_positives=false_positives
        )
        self.confusion_matrix.add(
            true_positives=true_positives,
            false_negatives=false_negatives,
            false_positives=false_positives,
        )
        return self.confusion_matrix
