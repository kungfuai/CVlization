from ..data_column import DataColumnType
from ..model_spec import ModelSpec, ModelInput, ModelTarget


def ObjectDetection(
    n_channels: int = 3, n_categories: int = 3, sequence_key: str = "detection"
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
            # TODO: take make_matching_figure from the following notebook into visulization sub folder.
            # https://colab.research.google.com/drive/17HPR0sz1iXu3wTw7b0p6BMqvRTDmfT-x?usp=sharing
        ]

    return ModelSpec(model_inputs=get_model_inputs(), model_targets=get_model_targets())
