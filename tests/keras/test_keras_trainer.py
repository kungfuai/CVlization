import logging
import pytest

from cvlization.tensorflow.keras_model_factory import KerasModelFactory
from cvlization.tensorflow.keras_trainer import KerasTrainer
from cvlization.tensorflow.encoder.keras_image_encoder import KerasImageEncoder
from cvlization.tensorflow.net.simple_conv_net import SimpleConvNet
from cvlization.tensorflow import sequence
from ..ml_dataset_utils import prepare_ml_datasets

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def create_model(model_inputs, model_targets):
    factory = KerasModelFactory(
        model_inputs=model_inputs,
        model_targets=model_targets,
        eager=True,
        image_encoder=KerasImageEncoder(
            backbone=SimpleConvNet((28, 28, 1)),
            pool_name="avg",
        ),
    )
    model_checkpoint = factory()
    model = model_checkpoint.model

    return model


def test_mnist_multiclass():
    train_data, val_data = prepare_ml_datasets()
    keras_ds = sequence.from_ml_dataset(train_data)
    first_batch = next(iter(keras_ds))
    x, y, _ = first_batch
    assert x is not None
    model = create_model(train_data.model_inputs, train_data.model_targets)
    trainer = KerasTrainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        epochs=3,
        use_multiprocessing=False,
        experiment_tracker=None,
    )
    trainer.train()


@pytest.mark.skip(
    reason="training with multiprocessing seems to incur significant overhead"
)
def test_mnist_multiclass_with_multiprocessing():
    train_data, val_data = prepare_ml_datasets()
    model = create_model(train_data.model_inputs, train_data.model_targets)
    trainer = KerasTrainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        epochs=2,
        use_multiprocessing=True,
    )
    trainer.train()


if __name__ == "__main__":
    test_mnist_multiclass()
    # test_mnist_multiclass_with_multiprocessing()
