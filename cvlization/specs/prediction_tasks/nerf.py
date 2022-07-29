from ..data_column import DataColumnType
from ..model_spec import ModelSpec, ModelInput, ModelTarget


def Nerf(n_channels: int = 3, channels_first: bool = False, **kwargs) -> ModelSpec:
    def get_model_targets():
        if channels_first:
            raw_shape = [n_channels, None, None]
        else:
            raw_shape = [None, None, n_channels]
        return [
            ModelTarget(
                key="image",
                column_type=DataColumnType.IMAGE,
                raw_shape=raw_shape,
            ),
        ]

    def get_model_inputs():
        return [
            # The following targets should have the same sequence length.
            ModelInput(
                key="pose",  # camera pose
                column_type=DataColumnType.NUMERICAL,
                raw_shape=[4, 4],
            ),
            ModelInput(
                key="focal",
                column_type=DataColumnType.NUMERICAL,
                raw_shape=[1],
            ),
            ModelInput(
                key="output_image_height",
                column_type=DataColumnType.NUMERICAL,
                raw_shape=[1],
            ),
            ModelInput(
                key="output_image_width",
                column_type=DataColumnType.NUMERICAL,
                raw_shape=[1],
            ),
        ]

    return ModelSpec(
        model_inputs=get_model_inputs(), model_targets=get_model_targets(), **kwargs
    )
