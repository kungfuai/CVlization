from tensorflow.keras import datasets
from tensorflow import keras
import numpy as np
import logging

from tensorflow.python import eager
from cvlization.data.ml_dataset import (
    DataRows,
    ModelInput,
    ModelTarget,
    MLDataset,
    DataColumnType,
)
from cvlization.keras.model_factory import KerasModelFactory
from cvlization.keras.trainer import KerasTrainer
from cvlization.keras.encoder.keras_image_encoder import KerasImageEncoder
from cvlization.keras.net.simple_conv_net import SimpleConvNet
from cvlization.losses.loss_type import LossType
from cvlization.keras.metrics.top_mistakes import TopMistakes
from cvlization.keras import sequence


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()


class MNISTDataRows(DataRows):
    # TODO: move to lib
    def __init__(self, *args, is_train=True, **kwargs):
        super().__init__(*args, **kwargs)
        limit = 200
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


def test_mnist_multiclass():
    model_inputs = [
        ModelInput(
            key="image", raw_shape=[28, 28, 1], column_type=DataColumnType.IMAGE
        ),
        ModelInput(
            key="digit_is_larger_than_2",
            raw_shape=[1],
            column_type=DataColumnType.BOOLEAN,
        ),
        # ModelInput(
        #     key="digit",
        #     raw_shape=[10],
        #     column_type=DataColumnType.NUMERICAL,
        # ),
    ]
    model_targets = [
        ModelTarget(
            key="digit",
            column_type=DataColumnType.CATEGORICAL,
            n_categories=10,
            loss=LossType.CATEGORICAL_CROSSENTROPY,
            # TODO: metric should be not concerned about particular ml framework
            metrics=[keras.metrics.CategoricalAccuracy()],
            # loss_weight=
            #
            # AttributeError: 'tuple' object has no attribute 'shape'
            # metrics=["accuracy"],
        ),
        ModelTarget(
            key="digit_is_even",
            column_type=DataColumnType.BOOLEAN,
            n_categories=2,
            loss=LossType.BINARY_CROSSENTROPY,
            metrics=[keras.metrics.AUC(), TopMistakes()],
        ),
    ]
    train_data = MLDataset(
        data_rows=MNISTDataRows(is_train=True),
        model_inputs=model_inputs,
        model_targets=model_targets,
        batch_size=16,
    )
    val_data = MLDataset(
        data_rows=MNISTDataRows(is_train=False),
        model_inputs=model_inputs,
        model_targets=model_targets,
        batch_size=16,
    )
    keras_seq = sequence.from_ml_dataset(train_data)
    first_batch = keras_seq[0]
    x, y, _ = first_batch
    # print([i.shape for i in x])
    # print(y)
    # print([i.shape for i in y])
    # raise
    create_model = KerasModelFactory(
        model_inputs=model_inputs,
        model_targets=model_targets,
        eager=False,
        image_encoder=KerasImageEncoder(
            trunk=SimpleConvNet((28, 28, 1)),
            pool_name=None,
        ),
    )
    model = create_model()
    print(model.summary())

    trainer = KerasTrainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        # train_steps_per_epoch=100,
        epochs=50,
    )
    trainer.train()


if __name__ == "__main__":
    test_mnist_multiclass()
