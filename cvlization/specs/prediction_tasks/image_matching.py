from ..data_column import DataColumnType
from ..model_spec import ModelSpec, ModelInput, ModelTarget


def ImageMatching(sequence_key: str = "keypoints") -> ModelSpec:
    """
    Example models: LoFTR
    """

    def get_model_inputs():
        return [
            ModelInput(key="image1", column_type=DataColumnType.IMAGE),
            ModelInput(key="image2", column_type=DataColumnType.IMAGE),
        ]

    def get_model_targets():
        return [
            # The following targets should have the same sequence length.
            ModelTarget(
                key="pts1",
                column_type=DataColumnType.KEYPOINTS,
                raw_shape=[None, 2],
                sequence=sequence_key,
            ),
            ModelTarget(
                key="pts2",
                column_type=DataColumnType.KEYPOINTS,
                raw_shape=[None, 2],
                sequence=sequence_key,
            ),
            ModelTarget(
                key="confidence",
                column_type=DataColumnType.NUMERICAL,
                raw_shape=[None, 1],
                sequence=sequence_key,
            ),
            # TODO: take make_matching_figure from the following notebook into visulization sub folder.
            # https://colab.research.google.com/drive/17HPR0sz1iXu3wTw7b0p6BMqvRTDmfT-x?usp=sharing
        ]

    return ModelSpec(model_inputs=get_model_inputs(), model_targets=get_model_targets())
