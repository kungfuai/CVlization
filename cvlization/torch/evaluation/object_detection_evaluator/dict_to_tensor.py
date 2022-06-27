import torch
from typing import Dict, List, Optional, Tuple


class DictToTensor:
    def _targets_to_tensor(self, targets: List[Dict[str,     torch.Tensor]]) -> List[Optional[    torch.Tensor]]:
        targets_tensors = []
        for target in targets:
            if len(target["boxes"]) > 0:
                target_tensor = torch.concat(
                    (
                        target["boxes"],
                        target["labels"].unsqueeze(dim=1),
                    ),
                    dim=1,
                )
                targets_tensors.append(target_tensor)
            else:
                targets_tensors.append(None)
        return targets_tensors

    def _detections_to_tensor(self, detections: List[Dict[str,     torch.Tensor]]) -> List[Optional[    torch.Tensor]]:
        detections_tensors = []
        for detection in detections:
            if len(detection["boxes"]) > 0:
                target_tensor = torch.concat(
                    (
                        detection["boxes"],
                        detection["labels"].unsqueeze(dim=1),
                        detection["scores"].unsqueeze(dim=1),
                    ),
                    dim=1,
                )
                detections_tensors.append(target_tensor)
            else:
                detections_tensors.append(None)
        return detections_tensors

    def __call__(self, targets: Optional[List[Dict[str,     torch.Tensor]]]=None, detections: Optional[List[Dict[str,     torch.Tensor]]]=None) -> Tuple[List[Optional[    torch.Tensor]], List[Optional[    torch.Tensor]]]:
        """
        Input:
            targets (List[dict]): List of targets in retinanet format
            detections (List[dict]): List of detections in retinanet format
        Output:
            targets (List[torch.Tensor]): List of (N, 5) shape tensors containing [[x_1, y_1, x_2, y_2, class_id]]
            detections (List[torch.Tensor]): List of (N, 6) shape tensors containing [[x_1, y_1, x_2, y_2, class_id, score]]
        """
        if targets and detections:
            targets = self._targets_to_tensor(targets=targets)
            detections = self._detections_to_tensor(detections=detections)
            return targets, detections
        if targets:
            targets = self._targets_to_tensor(targets=targets)
            return targets
        if detections:
            detections = self._detections_to_tensor(detections=detections)
            return detections
