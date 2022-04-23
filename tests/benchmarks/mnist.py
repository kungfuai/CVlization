import numpy as np
from tensorflow.keras import datasets

from cvlization.data.ml_dataset import (
    ModelInput,
    ModelTarget,
    DataColumnType,
    DataRows,
)
from cvlization.specs import LossType


class MNISTDataRows(DataRows):
    # TODO: make this class not dependent on tensorflow.
    def __init__(self, *args, is_train=True, limit=200, **kwargs):
        super().__init__(*args, **kwargs)

        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        self.x_train = x_train[:limit, :].astype(float) / 255.0
        self.y_train = y_train[:limit]
        self.x_test = x_test[:limit, :].astype(float) / 255.0
        self.y_test = y_test[:limit]
        self.is_train = is_train

    def __getitem__(self, i: int):
        if self.is_train:
            x, y = self.x_train, self.y_train
        else:
            x, y = self.x_test, self.y_test
        row = {
            "image": np.reshape(x[i, :], (28, 28, 1)),
            "digit": y[i],
            "digit_is_larger_than_2": y[i] > 2,
            "digit_is_even": y[i] % 2 == 0,
        }
        return row

    def __len__(self):
        if self.is_train:
            return len(self.y_train)
        else:
            return len(self.y_test)


model_inputs = [
    ModelInput(key="image", raw_shape=[28, 28, 1], column_type=DataColumnType.IMAGE),
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
    ),
    ModelTarget(
        key="digit_is_even",
        column_type=DataColumnType.BOOLEAN,
        n_categories=2,
        loss=LossType.BINARY_CROSSENTROPY,
    ),
]
