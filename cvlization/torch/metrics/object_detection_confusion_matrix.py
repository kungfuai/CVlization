import numpy as np
import torch
from torchvision.ops.boxes import box_iou, nms


class ObjectDetectionConfusionMatrix:
    """
    This class takes object detection results and produces a confusion matrix.

    Args:
        targets (List[Dict]): List of Dictionaries containing "boxes", "label"
        predictions (List[Dict]): List of Dictionaries containing "boxes", "label", and "scores"
        score_threshold (float): Minimum score required to be considered as a detection
        detection_iou_threshold (float): Minimum IOU overlap between ground truth and prediction boxes to be considered true positive

    Once instantiated, the class has the following attributes: self.true_pos, self.false_pos, self.false_neg, and self.score_threshold
    These attributes allow metrics to be calculated using the EvalMetrics class.
    """

    def __init__(
        self,
        targets=[],
        predictions=[],
        score_threshold: float = 0.20,
        detection_iou_threshold: float = 0.5,
    ):
        self.score_threshold = score_threshold
        self.detection_iou_threshold = detection_iou_threshold

        self.is_dict = False
        targets, predictions = self.make_lists(targets=targets, predictions=predictions)

        self.init_confusion_matrix()

        if not self.is_dict:
            self.calculate_confusion_matrix(targets=targets, predictions=predictions)
        if self.is_dict:
            self.calculate_ensemble_confusion_matrix(
                targets=targets, predictions=predictions
            )

    def init_confusion_matrix(self):
        self.true_pos = 0
        self.false_pos = 0
        self.false_neg = 0

    def make_lists(self, targets, predictions):
        if isinstance(targets, dict) and isinstance(predictions, dict):
            if "boxes" not in targets.keys() and "boxes" not in predictions.keys():
                self.is_dict = True
                for _, v in targets.items():
                    v = self.make_list(v)
                for _, v in predictions.items():
                    v = self.make_list(v)
        # elif not isinstance(targets, dict) and not isinstance(predictions, dict):
        if not self.is_dict:
            targets = self.make_list(targets)
            predictions = self.make_list(predictions)

        return targets, predictions

    def make_list(self, item):
        return [item] if not isinstance(item, list) else item

    def _filter_predictions_by_score(self, prediction, score_threshold):
        # TODO: need to implement actual filtering.
        keep_idx = prediction["scores"] >= score_threshold
        boxes_ = prediction["boxes"][keep_idx]
        return boxes_

    def filter_by_nms(self, prediction: torch.Tensor):
        """Performs NMS thresholding
        Args:
            prediction (Dict[Tensor]): Dictionary containing key "boxes" (shape = Tensor[N, 4]) and key "scores" (shape = Tensor[N])
        Returns:
            prediction = (Dict[Tensor]): Dictionary containing key "boxes" (shape = Tensor[N, 4]) and key "scores" (shape = Tensor[N])
        """
        keep_idx = nms(prediction["boxes"], prediction["scores"], self.nms_threshold)
        prediction["scores"] = prediction["scores"][keep_idx]
        prediction["boxes"] = prediction["boxes"][keep_idx]
        return prediction

    def calculate_confusion_matrix(self, targets, predictions):
        for i, prediction in enumerate(predictions):
            kept_boxes = self._filter_predictions_by_score(
                prediction, self.score_threshold
            )
            if kept_boxes.numel():
                if targets[i]["boxes"].numel():
                    ious = box_iou(targets[i]["boxes"], kept_boxes)
                    self.update_confusion_matrix(ious)
                else:
                    self.false_pos += len(kept_boxes)
            else:
                if targets[i]["boxes"].numel():
                    self.false_neg += len(targets[i]["boxes"])

    def calculate_ensemble_confusion_matrix(self, targets, predictions):
        length = self.get_length(targets)
        for p in range(length):
            gt_list = []
            dt_list = []
            for target, prediction in zip(targets.values(), predictions.values()):
                tar, pred = target[p], prediction[p]
                kept_boxes = self._filter_predictions_by_score(
                    pred, self.score_threshold
                )
                gt_list.append(tar["boxes"])
                dt_list.append(kept_boxes)
            self.update_ensemble_confusion_matrix(gt_list=gt_list, dt_list=dt_list)

    def update_ensemble_confusion_matrix(self, gt_list, dt_list):
        dts_exist = False
        iou_applied = False
        if dt_list:
            dts = self.ensemble_filter(dt_list=dt_list)
            if dts.numel():
                dts_exist = True
                if gt_list:
                    if gt_list[0].numel():
                        iou_applied = True
                        ious = box_iou(gt_list[0], dts)
                        self.update_confusion_matrix(ious)
            if dts_exist and not iou_applied:
                self.false_pos += len(dts)
        if not dts_exist:
            if gt_list:
                if gt_list[0].numel():
                    self.false_neg += len(gt_list[0])

    def ensemble_filter(self, dt_list):
        dts = dt_list[0]
        for i in range(len(dt_list) - 1):
            keep = []
            ious = box_iou(dts, dt_list[i + 1])
            for k in range(len(ious)):
                if ious[k].numel():
                    if torch.max(ious[k]) > self.detection_iou_threshold:
                        keep.append(k)
            if keep:
                dts = torch.stack([dts[k] for k in keep])
            else:
                dts = torch.Tensor(size=(0, 4)).double()
        return dts

    def get_length(self, targets):
        for key in targets.keys():
            length = len(targets[key])
        return length

    def update_confusion_matrix(self, ious):
        tps, fps, fns = self.get_values(ious)
        self.true_pos += tps
        self.false_pos += fps
        self.false_neg += fns

    def determine_detections(self, ious):
        tps, fps = [], []
        excluded_targets = []
        for prediction_idx, iou in enumerate(ious):
            c_iou = iou.copy()
            c_iou[excluded_targets] = 0  # i = pred, j = GT
            if c_iou.any():
                target_idx = np.argmax(c_iou)
                score = iou[target_idx]
                if score >= self.detection_iou_threshold:
                    tps.append((prediction_idx, target_idx))
                    excluded_targets.append(target_idx)
                else:
                    fps.append((prediction_idx, None))
            else:
                fps.append((prediction_idx, None))
        return tps, fps, excluded_targets

    def get_values(self, ious):
        ious = torch.swapaxes(ious, 0, 1)
        ious = ious.detach().numpy()
        tps, fps, excluded_targets = self.determine_detections(ious)
        fns = [(x, None) for x in np.arange(ious.shape[1]) if x not in excluded_targets]
        return len(tps), len(fps), len(fns)
