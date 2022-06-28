import torch
from .match_filter import MatchFilter
from typing import List, Optional, Tuple


class MatchDeterminer:
    """
    Used by BoxMatcher to create lists of matched and unmatched detections

    __call__ Args:
        iou_matrix_list (list[Tensor]): Contains tensor of IOU values between targets and detections (Jaccard index)
        scores_list (list[Tensor]): Scores corresponding to detections in the iou_matrix_list
        num_unmatched_fns (int): Number of unmatched targets that exist because there were NO detections made
                                for that image. This is used to construct match tensors that contain only "nan"

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

    def _make_matched_detections_table(
        self, iou_matrix: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        try:
            # Try to get the top 2 highest scoring detections
            table = self._get_top2_dets(iou_matrix=iou_matrix, scores=scores)
        except RuntimeError as e:
            # If there are less than two detections in total, try to get only the top 1 detection
            table = self._get_top_det(iou_matrix, scores=scores)
        return table

    def _get_top2_dets(
        self, iou_matrix: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """Returns Nx7 tensor containing: 1. Ground Truth box Index, 2. IOU of best detection box, 3. IOU of second best detection box
        4. Index of best detection box, 5. Index of second best detection box, 6. Score of best detection box, 7. Score of second best
        detection box."""
        gts = torch.arange(0, iou_matrix.shape[0]).unsqueeze(dim=0)
        top2 = torch.topk(iou_matrix, 2)
        maxes = torch.transpose(top2.values, 0, 1)
        maxes = torch.stack([maxes[0], maxes[1]])
        indices = torch.transpose(top2.indices, 0, 1)
        indices = torch.stack([indices[0], indices[1]])
        scores_to_add = scores[indices]
        return torch.concat([gts.int(), maxes, indices, scores_to_add], dim=0)

    def _get_top_det(self, iou_matrix, scores):
        """Same as get_top2_dets except the second best detection does not exist, and is thus assigned NAN"""
        gts = torch.arange(0, iou_matrix.shape[0]).unsqueeze(dim=0)
        num_targets = iou_matrix.shape[0]
        nan_tensor = torch.tensor([float("nan") for i in range(num_targets)])
        top1 = torch.max(iou_matrix, 1)
        maxes = torch.stack([top1.values, nan_tensor])
        scores_to_add = scores[top1.indices]
        scores_to_add = torch.stack([scores_to_add, nan_tensor])
        indices = torch.stack([top1.indices, nan_tensor])
        return torch.concat([gts.int(), maxes, indices, scores_to_add], dim=0)

    def _find_unmatched_detections(
        self, matched_detections: torch.Tensor, scores: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """If a detection is unmatched (false positive), we add its score to a tensor that is returned. This will allow
        us to later filter by score in order to count false positives for different score thresholds"""
        matched_detections_indices = matched_detections[2, :].tolist()
        unmatched_detections = []
        for i in range(len(scores)):
            if i not in matched_detections_indices:
                unmatched_detections.append(scores[i])
        unmatched_detections = [l for l in unmatched_detections if l != []]
        if len(unmatched_detections) == 0:
            unmatched_detections = None
        elif len(unmatched_detections) == 1:
            unmatched_detections = unmatched_detections[0].unsqueeze(dim=0)
        else:
            unmatched_detections = torch.stack(unmatched_detections, dim=0)
        return unmatched_detections

    def _add_empty_detections_tensor(
        self, matched_detections_list: List[torch.Tensor], num_unmatched_fns: int
    ) -> List[torch.Tensor]:
        """If there were no detections at all for a particular image, we can not use the get_top2_dets or the
        get_top_det method. Thus with this method, we create the NAN tensor we need"""
        unmatched_fns = [float("nan") for i in range(num_unmatched_fns)]
        unmatched_fns = torch.tensor(unmatched_fns).unsqueeze(dim=0)
        unmatched_fns = torch.concat(
            [
                torch.ones(size=(1, num_unmatched_fns)),
                unmatched_fns,
                unmatched_fns,
                unmatched_fns,
            ],
            dim=0,
        )
        matched_detections_list.append(unmatched_fns)
        return matched_detections_list

    def __call__(
        self,
        iou_matrix_list: List[torch.Tensor],
        scores_list: List[torch.Tensor],
        num_unmatched_fns: int,
    ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        """See documentation at top of class"""
        matched_detections_list = []
        unmatched_detections_list = []
        for i, iou_matrix in enumerate(iou_matrix_list):
            matched_detections = self._make_matched_detections_table(
                iou_matrix=iou_matrix, scores=scores_list[i]
            )
            matched_detections = MatchFilter.remove_double_scoring(
                matched_detections=matched_detections
            )
            matched_detections = MatchFilter.remove_lower_scoring(
                matched_detections=matched_detections
            )
            matched_detections = MatchFilter.remove_nans(
                matched_detections=matched_detections
            )
            matched_detections_list.append(matched_detections)

            unmatched_detections = self._find_unmatched_detections(
                matched_detections=matched_detections, scores=scores_list[i]
            )
            unmatched_detections_list.append(unmatched_detections)
        matched_detections_list = self._add_empty_detections_tensor(
            matched_detections_list=matched_detections_list,
            num_unmatched_fns=num_unmatched_fns,
        )
        return matched_detections_list, unmatched_detections_list
