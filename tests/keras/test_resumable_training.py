"""
ENV=prod python3 -m pytest src/tests/test_resumable_training.py
"""
import logging
import tensorflow as tf
import numpy as np
from unittest.mock import patch

from cvlization.tensorflow.keras_model_factory import KerasModelFactory
from cvlization.tensorflow.keras_trainer import KerasTrainer
from cvlization.tensorflow.encoder.keras_image_encoder import KerasImageEncoder
from cvlization.tensorflow.net.simple_conv_net import SimpleConvNet
from cvlization.tensorflow import sequence
from ..ml_dataset_utils import prepare_ml_datasets


LOGGER = logging.getLogger(__name__)


def create_model(model_inputs, model_targets, checkpoint_path=None) -> tf.keras.Model:
    factory = KerasModelFactory(
        model_inputs=model_inputs,
        model_targets=model_targets,
        eager=False,
        image_encoder=KerasImageEncoder(
            backbone=SimpleConvNet((28, 28, 1)),
            pool_name="avg",
        ),
        model_checkpoint_path=checkpoint_path,
        n_gradients=2,
    )
    model_checkpoint = factory()
    model = model_checkpoint.model
    return model


def test_training_can_resume(tmpdir):
    train_data, val_data = prepare_ml_datasets()
    keras_seq = sequence.from_ml_dataset(train_data)
    first_batch = keras_seq[0]
    x, y, _ = first_batch
    assert x is not None
    assert y is not None
    model_to_save = create_model(train_data.model_inputs, train_data.model_targets)
    model_checkpoint_path = tmpdir.join("checkpoint").join("model.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_checkpoint_path), save_best_only=True
        )
    ]
    trainer = KerasTrainer(
        model=model_to_save,
        train_dataset=train_data,
        val_dataset=val_data,
        epochs=2,
        use_multiprocessing=False,
        callbacks=callbacks,
        experiment_tracker=None,
    )
    trainer.train()

    # The number of rows in history.csv indicates how many epochs it actually ran through.
    tf.keras.backend.clear_session()
    LOGGER.info("** Resumed training starts...")

    assert model_checkpoint_path.isfile()
    resumed_model = create_model(
        train_data.model_inputs,
        train_data.model_targets,
        checkpoint_path=str(model_checkpoint_path),
    )

    assert len(model_to_save.trainable_variables) == len(
        resumed_model.trainable_variables
    )
    for v1, v2 in zip(
        model_to_save.trainable_variables, resumed_model.trainable_variables
    ):
        assert np.all(v1.numpy() == v2.numpy())
