from hashlib import new
import math
import torch
from torchvision.ops import box_iou, nms
from .match_determiner import MatchDeterminer
from typing import Any, List, Optional, Tuple, Union


class BoxMatcher:
    """
    This class matches detection boxes to target boxes for a given iou_detection_threshold

    __init__ Args:
        iou_detection_threshold (float): The minimum intersection-over-union (iou) for a detection to be counted
                                        as a true positive

    __call__ Args:
        targets (list[torch.Tensor]): list of tensors created by DictToTensor
        detections (list[torch.Tensor]): list of tensors created by DictToTensor

    __call__ Returns:
        matched (list[Tensor]): list of (N x 4) Tensors. Len(matched) = len(targets) = len(images). Every target
        is contained in the tensor (Ground Truth Box Index). If if it is a false negative (no matched detection), the
        rest of the values are NAN.
            Tensor contains:
                [[Ground Truth Box Index],
                [IOU Between Ground Truth and Detection],
                [Detection Box Index],
                [Score for Detecton]]

        unmatched (list[Tensor]): list of (1 x N) Tensors. Contains unmatched detections (False positives)
            Tensor contains:
            [[score of unmatched detection 0], [score of unmatched detection 1], [score of unmatched detection 2], [etc]]
    """

    def __init__(self, iou_detection_threshold: float = 0.5) -> None:
        self.iou_detection_threshold = iou_detection_threshold

    def _filter_by_nms(
        self, dt_boxes: torch.Tensor, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """If there are overlapping detection boxes, we discard the boxes with lower scores"""
        indices = nms(
            boxes=dt_boxes,
            scores=scores,
            iou_threshold=self.iou_detection_threshold,
        )
        indices = torch.sort(indices).values
        dt_boxes = torch.stack([dt_boxes[idx] for idx in indices])
        scores = torch.stack([scores[idx] for idx in indices])
        return dt_boxes, scores

    def _not_empty(self, lst):
        return True if len(lst) > 0 else False

    def _append_unmatched_dets(
        self, unmatched: List[Any], scores: torch.Tensor
    ) -> List[torch.Tensor]:
        if len(scores) > 1:
            unmatched.append(scores.squeeze())
        elif len(scores) == 1:
            unmatched.append(scores)
        return unmatched

    def _coerce_correct_shape(
        self, unmatched: List[Union[Any, torch.Tensor]]
    ) -> torch.Tensor:
        if len(unmatched) == 1:
            unmatched = unmatched[0]
        elif len(unmatched) > 1:
            unmatched = torch.concat(unmatched, dim=0)
        return unmatched

    def _calculate_iou_matrices(
        self,
        targets: List[Optional[torch.Tensor]],
        detections: List[Optional[torch.Tensor]],
    ) -> Union[
        Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, int],
        Tuple[List[torch.Tensor], List[torch.Tensor], List[Any], int],
    ]:
        iou_matrix_list, scores_list, unmatched = [], [], []
        num_unmatched_fns = 0
        for target, detection in zip(targets, detections):
            if not isinstance(detection, type(None)):
                if len(detection) == 0:
                    detection = None

            if not isinstance(target, type(None)):
                if len(target) == 0:
                    target = None

            if isinstance(detection, type(None)) and not isinstance(target, type(None)):
                num_unmatched_fns = len(target)

            if not isinstance(detection, type(None)) and isinstance(target, type(None)):
                unmatched = self._append_unmatched_dets(
                    unmatched=unmatched, scores=detection[:, -1]
                )
            if isinstance(detection, type(None)) and isinstance(target, type(None)):
                pass

            if not isinstance(detection, type(None)) and not isinstance(
                target, type(None)
            ):
                dt_boxes, scores = self._filter_by_nms(
                    dt_boxes=detection[:, 0:4], scores=detection[:, -1]
                )
                iou_matrix_list.append(box_iou(target[:, 0:4], dt_boxes))
                scores_list.append(scores)

        unmatched = self._coerce_correct_shape(unmatched=unmatched)

        return iou_matrix_list, scores_list, unmatched, num_unmatched_fns

    def __call__(
        self,
        targets: List[Optional[torch.Tensor]],
        detections: List[Optional[torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        """See documentation at top of class"""
        self.targets = targets
        self.detections = detections
        (
            iou_matrix_list,
            scores_list,
            initial_unmatched,
            num_unmatched_fns,
        ) = self._calculate_iou_matrices(targets, detections)
        matched, unmatched = MatchDeterminer()(
            iou_matrix_list=iou_matrix_list,
            scores_list=scores_list,
            num_unmatched_fns=num_unmatched_fns,
        )
        if len(initial_unmatched) > 0:
            unmatched.append(initial_unmatched)
        return matched, unmatched
