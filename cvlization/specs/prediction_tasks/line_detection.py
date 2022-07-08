from ..data_column import DataColumnType
from ..model_spec import ModelSpec, ModelInput, ModelTarget


def LineDetection(
    n_channels: int = 3,
    n_categories: int = 1,
    sequence_key: str = "detection",
    **kwargs
) -> ModelSpec:
    """
    Example models: RetinaNet, Pix2Seq
    """

    def get_model_inputs():
        return [
            ModelInput(
                key="image",
                column_type=DataColumnType.IMAGE,
                raw_shape=[None, None, n_channels],
            ),
        ]

    def get_model_targets():
        return [
            # The following targets should have the same sequence length.
            ModelTarget(
                key="lines",
                column_type=DataColumnType.LINES,
                raw_shape=[None, 4],
                sequence=sequence_key,
            ),
            # TODO: add line_labels
            ModelTarget(
                key="labels",
                column_type=DataColumnType.CATEGORICAL,
                raw_shape=[None, 1],
                sequence=sequence_key,
                n_categories=n_categories,
            ),
        ]

    return ModelSpec(
        model_inputs=get_model_inputs(), model_targets=get_model_targets(), **kwargs
    )
