from dataclasses import dataclass
from typing import Optional, List
import logging
from multiprocessing import cpu_count

from tensorflow import keras
from . import sequence


from ..data import MLDataset

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@dataclass
class KerasTrainer:
    """Trainer does not care about internal details of the model or datasets.

    It performs quality checks in the data flow.
    It configures the optimizer, metrics, losses.
    It saves the trainable parameters of the whole pipeline.
    ? It tracks params, metrics, artifacts.

    # TODO: expose easy access to image encoding output
    """

    model: keras.Model
    train_dataset: MLDataset
    val_dataset: MLDataset
    epochs: Optional[int] = 10
    train_steps_per_epoch: Optional[int] = None
    val_steps_per_epoch: Optional[int] = None
    callbacks: Optional[List] = None

    def __post_init__(self):
        if self.model is None:
            raise ValueError("Please set the model field to a keras Model.")

    def _training_loop(self):
        train_data = sequence.from_ml_dataset(self.train_dataset)
        val_data = sequence.from_ml_dataset(self.val_dataset)
        self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.epochs,
            steps_per_epoch=self.train_steps_per_epoch,
            validation_steps=self.val_steps_per_epoch,
            workers=1,  # ,cpu_count(),
            use_multiprocessing=False,
            max_queue_size=100,
            callbacks=self.callbacks,
        )

    def train(self):
        self._train_val_datasets_should_have_matching_inputs_and_targets()
        self._training_data_stats()
        self._training_loop()

    def run(self):
        self.train()

    def _training_data_stats(self):
        train_seq = sequence.from_ml_dataset(self.train_dataset)
        for j, (inputs, targets, sample_weight) in enumerate(train_seq):
            LOGGER.info(f"batch {j}: {len(inputs[0])} examples")
            for i, input_array in enumerate(inputs):
                LOGGER.info(
                    f"{self.train_dataset.model_inputs[i].key}: mean={input_array.mean()}"
                )
            for i, target_array in enumerate(targets):
                LOGGER.info(
                    f"{self.train_dataset.model_targets[i].key}: mean={target_array.mean()}"
                )
                LOGGER.info(target_array)
            LOGGER.info("sample weights")
            LOGGER.info(sample_weight)
            if j >= 3:
                break

    def _train_val_datasets_should_have_matching_inputs_and_targets(self):
        assert len(self.train_dataset.model_inputs) == len(
            self.val_dataset.model_inputs
        )
        assert len(self.train_dataset.model_targets) == len(
            self.val_dataset.model_targets
        )
