import numpy as np
import torch
from .box_matcher import BoxMatcher
from .match_holder import MatchHolder
from .metrics import Metrics
from .dict_to_tensor import DictToTensor
from .pr_calculator import PRCalculator
from src.dataset.target import Target
from src.dataset.prediction import Prediction
from numpy import float64, ndarray
from typing import Dict, List, Optional, Union


class Evaluator:
    """
    Lists of targets and detections are added with the "add" method.
    The calculate method then calcualtes metrics based upon the targets and detections that were added.
    """

    def __init__(
        self,
        num_classes: int = 1,
        iou_detection_threshold: float = 0.5,
    ) -> None:
        self.num_classes = num_classes
        self.iou_detection_threshold = iou_detection_threshold

        self.box_matcher = BoxMatcher(iou_detection_threshold=iou_detection_threshold)
        self.match_holder = MatchHolder(num_classes=num_classes)
        self.metrics = Metrics()
        self.dict_to_tensor = DictToTensor()
        self.pr_calculator = PRCalculator(
            iou_detection_threshold=iou_detection_threshold
        )

    def _filter_nans(
        self, arr_list: List[Optional[torch.Tensor]]
    ) -> List[Optional[torch.Tensor]]:
        """Sometimes NANs are returned during PyTorch's 'Sanity Validation Check'. We need to eliminate these
        to avoid runtime errors that stop training before epoch 0 even starts"""
        filtered = []
        for arr in arr_list:
            if not isinstance(arr, type(None)):
                filtered.append(arr[~torch.any(arr.isnan(), dim=1)])
            else:
                filtered.append(arr)
        return filtered

    def _class_filter(
        self, arr_list: List[Optional[torch.Tensor]], class_id: int
    ) -> List[Optional[torch.Tensor]]:
        """Remove all targets/detections that are not for class_id"""
        filtered = []
        for arr in arr_list:
            if not isinstance(arr, type(None)):
                filtered_arr = arr[arr[:, 4] == class_id]
                filtered.append(filtered_arr)
            else:
                filtered.append(arr)
        return filtered

    def add(
        self,
        targets: List[Union[Target, Dict[str, torch.Tensor]]],
        detections: List[Union[Prediction, Dict[str, torch.Tensor]]],
    ) -> None:
        """Takes in a list (batch) of targets and detections (each item in the corresponds to a different image), determines matched
        and unmatched targets and detections, and stores these in the MatchHolder (self.match_holder). After all evaluation batches
        have been added, metrics will be calculated from the lists held in MatchHolder.

        Args:
            targets (List[Dict[str, torch.Tensor]]): _description_
            detections (List[Dict[str, torch.Tensor]]): _description_
        """
        self.metrics._num_images += len(targets)

        targets = Target.to_retinanet_dicts_list(targets)
        # TODO: Need to rename detection to prediction to be congruent with Object name
        detections = Prediction.to_retinanet_dicts_list(detections)

        targets, detections = self.dict_to_tensor(
            targets=targets, detections=detections
        )
        targets, detections = self._filter_nans(targets), self._filter_nans(detections)
        for i in range(self.num_classes):
            filtered_targets, filtered_detections = self._class_filter(
                targets, class_id=i
            ), self._class_filter(detections, class_id=i)
            matched_batch, unmatched_batch = self.box_matcher(
                targets=filtered_targets, detections=filtered_detections
            )
            self.match_holder.add(
                matched=matched_batch, unmatched=unmatched_batch, class_id=i
            )

    def reset(self):
        self.metrics.reset()
        self.match_holder.reset()

    def calculate(self) -> None:
        pr_array_list = []

        # Each holder contains matched/unmatched lists for a single class. This loop lets us get class metrics
        for holder in self.match_holder.holder_list:
            matched_list, unmatched_list = holder.matched_list, holder.unmatched_list
            precision_recall_array = self.pr_calculator(
                matched_list=matched_list, unmatched_list=unmatched_list
            )
            pr_array_list.append(precision_recall_array)

        # A precision_recall_array is also calculated for all classes together. This will be used to calculate F1 score
        # F1 score is thus only calculated for all classes together and not for individual classes. This is done because the
        # F1_max occurs at a different score (confidence) threshold for each class. If we calculated F1 score individually for
        # each class, it would be a fallacy to average them for this reason
        all_classes_precision_recall_array = (
            self._get_all_classes_precision_recall_array()
        )

        # Calculate metrics from the precision_recall_arrays and update the metrics being stored as attributes
        # in the Metrics object (self.metrics)
        self.metrics.update_metrics(
            pr_array_list=pr_array_list,
            all_classes_precision_recall_array=all_classes_precision_recall_array,
        )

    def _get_all_classes_precision_recall_array(self) -> ndarray:
        all_matched = []
        all_unmatched = []
        for holder in self.match_holder.holder_list:
            all_matched.extend(holder.matched_list)
            all_unmatched.extend(holder.unmatched_list)

        all_classes_precision_recall_array = self.pr_calculator(
            matched_list=all_matched, unmatched_list=all_unmatched
        )
        return all_classes_precision_recall_array