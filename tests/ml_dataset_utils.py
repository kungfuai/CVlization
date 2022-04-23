from cvlization.data.ml_dataset import (
    ModelInput,
    ModelTarget,
    MLDataset,
    DataColumnType,
)
from cvlization.specs import EnsembleModelTarget, LossType, MetricType

from tests.benchmarks.mnist import MNISTDataRows


def prepare_ml_datasets():
    model_inputs = [
        ModelInput(
            key="image",
            raw_shape=[28, 28, 1],
            column_type=DataColumnType.IMAGE,
        ),
        ModelInput(
            key="image",
            input_groups=["default", "image_group1"],
            raw_shape=[28, 28, 1],
            column_type=DataColumnType.IMAGE,
        ),
        ModelInput(
            key="image",
            input_groups=["default", "image_group2"],
            raw_shape=[28, 28, 1],
            column_type=DataColumnType.IMAGE,
        ),
        ModelInput(
            key="digit_is_larger_than_2",
            raw_shape=[1],
            column_type=DataColumnType.BOOLEAN,
        ),
    ]
    model_targets = [
        ModelTarget(
            key="digit",
            column_type=DataColumnType.CATEGORICAL,
            n_categories=10,
            loss=LossType.CATEGORICAL_CROSSENTROPY,
            target_groups=["target_group1"],
            metrics=[MetricType.ACCURACY],
            prefer_logits=True,
            # metrics=["accuracy"],  # this does not work. AttributeError: 'tuple' object has no attribute 'shape'.
        ),
        ModelTarget(
            key="digit",
            column_type=DataColumnType.CATEGORICAL,
            n_categories=10,
            input_group="image_group1",
            target_groups=["target_group1"],
            loss=LossType.CATEGORICAL_CROSSENTROPY,
            metrics=[MetricType.ACCURACY],
        ),
        ModelTarget(
            key="digit_is_even",
            column_type=DataColumnType.BOOLEAN,
            n_categories=2,
            loss=LossType.BINARY_CROSSENTROPY,
            metrics=[MetricType.AUROC],
            loss_weight=0,
        ),
        EnsembleModelTarget(
            key="digit",
            column_type=DataColumnType.CATEGORICAL,
            n_categories=10,
            loss=LossType.CATEGORICAL_CROSSENTROPY,
            metrics=[MetricType.ACCURACY],
            target_group_to_ensemble="target_group1",
            loss_weight=0,
        ),
    ]
    train_data = MLDataset(
        data_rows=MNISTDataRows(is_train=True),
        model_inputs=model_inputs,
        model_targets=model_targets,
        batch_size=32,
        name="train_dataset",
    )
    val_data = MLDataset(
        data_rows=MNISTDataRows(is_train=False),
        model_inputs=model_inputs,
        model_targets=model_targets,
        batch_size=32,
        name="val_dataset",
    )
    return train_data, val_data
