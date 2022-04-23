from dataclasses import dataclass, field
from typing import Dict, List

from .. import ModelSpec
from ..specs import LossType, MetricType, ModelInput, ModelTarget, DataColumnType


@dataclass
class ImageClassification(ModelSpec):
    n_classes: int = 10
    num_channels: int = 3
    image_height: int = None
    image_width: int = None
    channels_first: bool = False

    def get_model_inputs(self):
        if self.channels_first:
            raw_shape = [self.num_channels, self.image_height, self.image_width]
        else:
            raw_shape = [self.image_height, self.image_width, self.num_channels]
        return [
            ModelInput(
                key="image",
                # Raw shape is the shape before transformation.
                raw_shape=raw_shape,
                column_type=DataColumnType.IMAGE,
            ),
        ]

    def get_model_targets(self):
        if self.n_classes == 2:
            return [
                ModelTarget(
                    key="label",
                    raw_shape=[1],
                    column_type=DataColumnType.BOOLEAN,
                    metrics=[MetricType.AUROC],
                    prefer_logits=True,
                )
            ]
        else:
            return [
                ModelTarget(
                    key="label",
                    raw_shape=[1],
                    column_type=DataColumnType.CATEGORICAL,
                    n_categories=self.n_classes,
                    loss=LossType.SPARSE_CATEGORICAL_CROSSENTROPY,
                    metrics=[MetricType.ACCURACY],
                    prefer_logits=True,
                )
            ]
