from typing import List, Callable, Optional, Union, Dict, Any
from dataclasses import dataclass, field as dataclass_field
import logging

from tensorflow import keras
from . import sequence


from ..data import MLDataset

# tf.autograph.set_verbosity(3)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@dataclass
class KerasTrainer:
    """Trainer does not care about internal details of the model or datasets.

    It performs quality checks in the data flow.
    It configures the optimizer, metrics, losses.
    It saves the trainable parameters of the whole pipeline.
    ? It tracks params, metrics, artifacts.

    # TODO: expose easy access to image encoding
    """

    model: keras.Model
    train_dataset: MLDataset
    val_dataset: MLDataset
    epochs: Optional[int] = 10
    train_steps_per_epoch: Optional[int] = None
    val_steps_per_epoch: Optional[int] = None

    # @pydantic.validator("model")
    # def model_must_be_from_functional_api(cls, v):
    #     return True

    class Config:
        arbitrary_types_allowed = True

    def _training_loop(self):
        train_data = sequence.from_ml_dataset(self.train_dataset)
        val_data = sequence.from_ml_dataset(self.val_dataset)
        self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.epochs,
            steps_per_epoch=self.train_steps_per_epoch,
            validation_steps=self.val_steps_per_epoch,
        )

    def train(self):
        self._train_val_datasets_should_have_matching_inputs_and_targets()
        self._training_data_stats()
        self._training_loop()

    def _training_data_stats(self):
        train_seq = sequence.from_ml_dataset(self.train_dataset)
        for j, (inputs, targets) in enumerate(train_seq):
            print("batch", j)
            for i, input_array in enumerate(inputs):
                print(
                    f"{self.train_dataset.model_inputs[i].key}: mean={input_array.mean()}"
                )
            for i, target_array in enumerate(targets):
                print(
                    f"{self.train_dataset.model_targets[i].key}: mean={target_array.mean()}"
                )
            if j >= 5:
                break

    def _train_val_datasets_should_have_matching_inputs_and_targets(self):
        assert len(self.train_dataset.model_inputs) == len(
            self.val_dataset.model_inputs
        )
        assert len(self.train_dataset.model_targets) == len(
            self.val_dataset.model_targets
        )
