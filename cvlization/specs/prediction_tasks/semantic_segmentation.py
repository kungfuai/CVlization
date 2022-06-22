from ..data_column import DataColumnType
from ..model_spec import ModelSpec, ModelInput, ModelTarget


def PanopticSegmentation(
    n_channels: int = 3,
    n_categories: int = 3,
    sequence_key: str = "detection",
    **kwargs
) -> ModelSpec:
    """
    Example models:
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
                key="stuff_mask",
                column_type=DataColumnType.IMAGE,
                raw_shape=[None, None, None],
                sequence=sequence_key,
            ),
        ]

    return ModelSpec(
        model_inputs=get_model_inputs(), model_targets=get_model_targets(), **kwargs
    )
