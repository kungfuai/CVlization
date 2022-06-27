import numpy as np
import torch
from src.evaluation.confusion_matrix_maker import ConfusionMatrixMaker
from src.evaluation.box_matcher import BoxMatcher
from src.evaluation.match_holder import MatchHolder
from src.evaluation.metrics import Metrics
from src.evaluation.dict_to_tensor import DictToTensor
from src.dataset.target import Target
from src.dataset.prediction import Prediction
from numpy import float64, ndarray
from src.evaluation.confusion_matrix import ConfusionMatrix
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
        score_thresholds: ndarray = np.arange(0, 1.00, 0.01),
    ) -> None:
        self.num_classes = num_classes
        self.iou_detection_threshold = iou_detection_threshold
        self.score_thresholds = score_thresholds

        self.box_matcher = BoxMatcher(iou_detection_threshold=iou_detection_threshold)
        self.match_holder = MatchHolder(num_classes=num_classes)
        self.metrics = Metrics(score_thresholds=score_thresholds)
        self.dict_to_tensor = DictToTensor()

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

    def calculate(self, recall_thresholds: ndarray = np.arange(0, 1, 0.01)) -> None:
        pr_array_list = []

        # Each holder contains matched/unmatched lists for a single class. This loop lets us get class metrics
        for holder in self.match_holder.holder_list:
            matched_list, unmatched_list = holder.matched_list, holder.unmatched_list
            precision_recall_array = np.zeros(shape=(4, len(self.score_thresholds)))

            # For each score threshold, calculate precision and recall and add it to the precision_recall_array
            for i, score_threshold in enumerate(self.score_thresholds):
                precision_recall_array = self._update_precision_recall_array(
                    matched_list=matched_list,
                    unmatched_list=unmatched_list,
                    iou_detection_threshold=self.iou_detection_threshold,
                    score_threshold=score_threshold,
                    precision_recall_array=precision_recall_array,
                    i=i,
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
            recall_thresholds=recall_thresholds,
        )

    def _get_all_classes_precision_recall_array(self) -> ndarray:
        all_classes_precision_recall_array = np.zeros(
            shape=(4, len(self.score_thresholds))
        )
        all_matched = []
        all_unmatched = []
        for holder in self.match_holder.holder_list:
            all_matched.extend(holder.matched_list)
            all_unmatched.extend(holder.unmatched_list)

        for i, score_threshold in enumerate(self.score_thresholds):
            all_classes_precision_recall_array = self._update_precision_recall_array(
                matched_list=all_matched,
                unmatched_list=all_unmatched,
                iou_detection_threshold=self.iou_detection_threshold,
                score_threshold=score_threshold,
                precision_recall_array=all_classes_precision_recall_array,
                i=i,
            )
        return all_classes_precision_recall_array

    def _update_precision_recall_array(
        self,
        matched_list: List[torch.Tensor],
        unmatched_list: List[Optional[torch.Tensor]],
        iou_detection_threshold: float,
        score_threshold: float64,
        precision_recall_array: ndarray,
        i: int,
    ) -> ndarray:
        """
        Returns a (3xN) array where:
            row 0 = precision
            row 1 = recall
            row 2 = f1 score
            row 3 = score (at which precision and recall were calculated)

        Each time this method is called it extends the array with the precision and recall values
        for the score_threshold that was past

        Note: F1 score is being calculated individual classes as well and stored in the precision_recall_array,
        however the F1 score for individual classes is not being reported in the Metrics object (unless self.num_classes = 1)
        """
        confusion_matrix = self._make_confusion_matrix(
            matched_list=matched_list,
            unmatched_list=unmatched_list,
            iou_detection_threshold=iou_detection_threshold,
            score_threshold=score_threshold,
        )
        precision_recall_array[0, i] = confusion_matrix.precision
        precision_recall_array[1, i] = confusion_matrix.recall
        precision_recall_array[2, i] = self._calculate_f1(
            precision=confusion_matrix.precision, recall=confusion_matrix.recall
        )
        precision_recall_array[3, i] = score_threshold
        return precision_recall_array

    def _calculate_f1(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0
        else:
            return 2 * (precision * recall) / (precision + recall)

    def _make_confusion_matrix(
        self,
        matched_list: List[torch.Tensor],
        unmatched_list: List[Optional[torch.Tensor]],
        iou_detection_threshold: float,
        score_threshold: float64,
    ) -> ConfusionMatrix:
        """Confusion matrix is the common name for a table holding the number of false positives, false negatives,
        true positives, and true negatives. In object detection, we do not count true negatives (they are hard to define
        so we dont do it)."""
        confusion_matrix_maker = ConfusionMatrixMaker(
            iou_detection_threshold=iou_detection_threshold,
            score_threshold=score_threshold,
        )
        return confusion_matrix_maker(
            matched_list=matched_list, unmatched_list=unmatched_list
        )
