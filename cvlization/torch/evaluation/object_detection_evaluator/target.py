import numpy as np
import warnings
from src.dataset.bbox import Bbox
from typing import List
from src.dataset.target_base import TargetBase

FORMATS = ["COCO", "Pascal_VOC"]

class Target(TargetBase):
    def __init__(self, bboxes: List[Bbox]) -> None:
        self.bboxes = bboxes

    @classmethod
    def to_retinanet_dicts_list(cls, batch):
        if isinstance(batch[0], Target):
            new_batch = []
            for b in batch:
                new_batch.append(b.to_retinanet_dict)
            return new_batch
        else:
            return batch

    def to_retinanet_dict(self):
        temp = {}
        temp["boxes"] = self.bboxes_as_torch(box_format="COCO")
        temp["labels"] = self.labels_as_torch
        temp["names"] = self.names

        target = {}
        for k, v in temp.items():
            if not any(x is None for x in v):
                target[k] = v
        return target

    @classmethod
    def from_retinanet_dict(cls, retinanet_dict):
        label_map = {
            "boxes": "bbox",
            "labels": "class_id",
            "names": "class_name",
            "scores": "score",
        }

        bboxes = []
        for i in range(len(tuple(retinanet_dict.values())[0])):
            try:
                bbox_dict = {}
                for k, v in retinanet_dict.items():
                    bbox_dict[label_map[k]] = v[i]
                bboxes.append(Bbox(**bbox_dict))
            except KeyError:
                warnings.warn(
                    f"Warning: The key {k} is not a standard key. Please change the key name to 'boxes', 'labels', 'names', or 'scores', or if the key is not needed, ignore this message."
                )
        return Target(bboxes=bboxes)