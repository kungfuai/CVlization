from ..data_column import DataColumnType
from ..model_spec import ModelSpec, ModelInput, ModelTarget


def InstanceSegmentation(
    n_channels: int = 3,
    n_categories: int = 3,
    sequence_key: str = "instance_seg",
    **kwargs
) -> ModelSpec:
    """
    Example models: MaskRCNN.
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
                key="bbox",
                column_type=DataColumnType.BOUNDING_BOXES,
                raw_shape=[None, 4],
                sequence=sequence_key,
            ),
            ModelTarget(
                key="bbox_class",
                column_type=DataColumnType.CATEGORICAL,
                raw_shape=[None, 1],
                n_categories=n_categories,
                sequence=sequence_key,
            ),
            ModelTarget(
                key="bbox_mask",
                column_type=DataColumnType.IMAGE,
                raw_shape=[None, None, None],
                # TODO: the n_channels of mask should be the same as num_categories.
                sequence=sequence_key,
            ),
        ]

    return ModelSpec(
        model_inputs=get_model_inputs(), model_targets=get_model_targets(), **kwargs
    )
