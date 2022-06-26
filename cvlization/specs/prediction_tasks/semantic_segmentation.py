from ..data_column import DataColumnType
from ..model_spec import ModelSpec, ModelInput, ModelTarget


def SemanticSegmentation(
    n_channels: int = 3,
    sequence_key: str = "detection",
    channels_first: bool = True,
    **kwargs
) -> ModelSpec:
    """
    Example models:
    """

    def get_model_inputs():
        image_shape = (
            [n_channels, None, None] if channels_first else [None, None, n_channels]
        )
        return [
            ModelInput(
                key="image",
                column_type=DataColumnType.IMAGE,
                raw_shape=image_shape,
            ),
        ]

    def get_model_targets():
        return [
            # The following targets should have the same sequence length.
            ModelTarget(
                key="stuff_mask",
                column_type=DataColumnType.MASK,
                raw_shape=[None, None, None],
                sequence=sequence_key,
            ),
        ]

    return ModelSpec(
        model_inputs=get_model_inputs(), model_targets=get_model_targets(), **kwargs
    )
