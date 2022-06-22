from ..model_spec import ModelSpec, ModelInput, ModelTarget
from ...specs import LossType, MetricType, DataColumnType


def ImageClassification(
    n_classes: int = 10,
    num_channels: int = 3,
    image_height: int = None,
    image_width: int = None,
    channels_first: bool = True,
    **kwargs,
) -> ModelSpec:
    """
    channels_first: whether the input image has the channels axis in the first dimension.
        For Pytorch, this is typically True. For Tensorflow, this is usually False.
    """

    def get_model_inputs():
        if channels_first:
            raw_shape = [num_channels, image_height, image_width]
        else:
            raw_shape = [image_height, image_width, num_channels]
        return [
            ModelInput(
                key="image",
                # Raw shape is the shape before transformation.
                raw_shape=raw_shape,
                column_type=DataColumnType.IMAGE,
            ),
        ]

    def get_model_targets():
        if n_classes == 2:
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
                    n_categories=n_classes,
                    loss=LossType.SPARSE_CATEGORICAL_CROSSENTROPY,
                    metrics=[MetricType.ACCURACY],
                    prefer_logits=True,
                )
            ]

    return ModelSpec(
        model_inputs=get_model_inputs(), model_targets=get_model_targets(), **kwargs
    )
