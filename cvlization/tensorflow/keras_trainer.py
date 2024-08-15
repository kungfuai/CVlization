from dataclasses import dataclass

from typing import Optional, List, Any, Union
import logging
import os
import mlflow
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from subprocess import check_output

from . import sequence
from ..specs import DataColumnType
from ..data import MLDataset
from ..base_trainer import BaseTrainer
from .callbacks.lr_finder_callback import LRFinderCallback


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@dataclass
class KerasTrainer(BaseTrainer):
    """Trainer does not care about internal details of the model or datasets.

    It performs quality checks in the data flow.
    It configures the optimizer, metrics, losses.
    It saves the trainable parameters of the whole pipeline.
    ? It tracks params, metrics, artifacts.

    # TODO: expose easy access to image encoding output
    """

    model: keras.Model
    train_dataset: Union[MLDataset, tf.data.Dataset]
    val_dataset: Union[MLDataset, tf.data.Dataset]
    epochs: Optional[int] = 10
    initial_epoch: Optional[int] = 0
    train_steps_per_epoch: Optional[int] = None
    val_steps_per_epoch: Optional[int] = None
    callbacks: Optional[List] = None
    use_lr_finder: Optional[bool] = False

    name: Optional[str] = "keras_trainer"
    experiment_tracker: Optional[Any] = mlflow
    use_multiprocessing: Optional[bool] = True
    workers: Optional[int] = cpu_count()
    max_queue_size: Optional[int] = 100
    log_input_images: Optional[bool] = False
    evaluate_before_train: Optional[bool] = False

    def __post_init__(self):
        super().__init__(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            experiment_tracker=self.experiment_tracker,
            log_input_images=self.log_input_images,
        )
        if self.model is None:
            raise ValueError("Please set the model field to a keras Model.")

    def _training_loop(self):
        if isinstance(self.train_dataset, MLDataset):
            train_data = sequence.from_ml_dataset(self.train_dataset)
        elif isinstance(self.train_dataset, tf.data.Dataset):
            train_data = self.train_dataset
        else:
            train_data = self.train_dataset
            LOGGER.warning(
                f"train_dataset is expected to be either a MLDataset or a tf.data.Dataset. Got {type(self.train_dataset)}"
            )
        if isinstance(self.val_dataset, MLDataset):
            val_data = sequence.from_ml_dataset(self.val_dataset)
        elif isinstance(self.val_dataset, tf.data.Dataset):
            val_data = self.val_dataset
        else:
            val_data = self.val_dataset
            LOGGER.warning(
                f"val_dataset is expected to be either a MLDataset or a tf.data.Dataset. Got {type(self.val_dataset)}"
            )

        if self.use_lr_finder:
            self.model.optimizer.lr = self._find_best_lr(train_data, val_data)

        LOGGER.info("Begin to fit model...")
        self.fit(train_data, val_data, self.epochs, self.callbacks)

    def _find_best_lr(self, train_data, val_data) -> float:
        LOGGER.info("LR Finder at work...")
        lrfinder = LRFinderCallback(min_lr=1e-6, max_lr=0.01)
        self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=1,
            steps_per_epoch=min(100, self.train_steps_per_epoch)
            if self.train_steps_per_epoch
            else None,
            validation_steps=min(20, self.val_steps_per_epoch)
            if self.val_steps_per_epoch
            else None,
            workers=self.workers,
            use_multiprocessing=self.use_multiprocessing,
            max_queue_size=self.max_queue_size,
            callbacks=[lrfinder],
        )
        best_lr = lrfinder.best_lr()
        LOGGER.info(
            f"Best Learning Rate: {best_lr}. Setting the starting lr in this optimizer."
        )

    def fit(self, train_data, val_data, epochs, callbacks):
        # Evaluate before training starts.
        if self.evaluate_before_train:
            LOGGER.info("First, evaluate the starting model...")
            metrics_and_losses = self.model.evaluate(
                val_data,
                steps=self.val_steps_per_epoch,
                return_dict=True,
                workers=self.workers,
                use_multiprocessing=self.use_multiprocessing,
                max_queue_size=self.max_queue_size,
            )
            if self.experiment_tracker:
                for metric_name, metric_value in metrics_and_losses.items():
                    self.experiment_tracker.log_metric(
                        "eval_" + metric_name, metric_value
                    )
                    LOGGER.info(f"metric eval_{metric_name}: {metric_value}")
            LOGGER.info("Done evaluate the starting model.")

        self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            initial_epoch=self.initial_epoch,
            steps_per_epoch=self.train_steps_per_epoch,
            validation_steps=self.val_steps_per_epoch,
            # workers=self.workers,
            # use_multiprocessing=self.use_multiprocessing,
            # max_queue_size=self.max_queue_size,
            callbacks=callbacks,
        )

    def _get_current_epoch(self, checkpoint_dir: str) -> int:
        df = pd.read_csv(os.path.join(checkpoint_dir, "history.csv"))
        LOGGER.info(f"history csv: {df}")
        LOGGER.info(f"{df.columns.tolist()}")
        return df.epoch.max()

    def _log_input_images(
        self,
        ds: Union[MLDataset, tf.data.Dataset],
        num_batches: int = 10,
        artifact_dir: str = "batches_train",
    ):
        # TODO(refactor): The functionality of fetching example images should be implemented by the dataset.
        # TODO: there should be a wrapper around tf.data.Dataset that maintains model_inputs, model_targets.
        LOGGER.info(f"Logging input images in {artifact_dir}...")
        if isinstance(ds, MLDataset):
            self._log_input_images_with_ml_dataset(ds, num_batches, artifact_dir)
        elif isinstance(ds, tf.data.Dataset):
            self._log_input_images_with_tf_dataset(ds, num_batches, artifact_dir)
        else:
            raise ValueError(
                f"To log input images, the dataset must be either a MLDataset or a tf.data.Dataset. Got {type(ds)}"
            )

    def _log_input_images_with_tf_dataset(
        self,
        ds: tf.data.Dataset,
        num_batches: int,
        artifacts_dir: str = "batches_train",
    ):
        LOGGER.info("logging images for tf dataset")
        # TODO: create a TFDataset class that inherits tf.data.Dataset.
        if not hasattr(ds, "model_inputs") or not hasattr(ds, "model_targets"):
            LOGGER.error(
                "Logging input images is not implemented for tf.data.Dataset. model_inputs and model_targets are required"
            )
            return
        first_model_target = ds.model_targets[0]
        if first_model_target.column_type == DataColumnType.BOOLEAN or (
            first_model_target.column_type == DataColumnType.CATEGORICAL
            and first_model_target.n_categories == 2
        ):
            should_add_target_value_to_filename = True
        else:
            should_add_target_value_to_filename = False

        for i, batch in enumerate(ds.take(num_batches)):
            LOGGER.info(f"to log images for batch {i}")
            input_tensors = batch[0]
            target_tensors = batch[1]
            first_target_tensor = target_tensors[0]
            for model_input, input_tensor in zip(ds.model_inputs, input_tensors):
                if model_input.column_type == DataColumnType.IMAGE:
                    LOGGER.info(f"to log images for {model_input.key}")
                    if should_add_target_value_to_filename:
                        target_value = first_target_tensor.numpy()[0]
                        if isinstance(target_value, list) or isinstance(
                            target_value, np.ndarray
                        ):
                            target_value = target_value[-1]
                        filepath = f"/tmp/{self.name}/label{target_value}_example{i}_{model_input.key}.png"
                    else:
                        filepath = f"/tmp/{self.name}/example{i}_{model_input.key}.png"
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    self._save_image(input_tensor.numpy()[0], filepath)
                    LOGGER.info("------------------------------")
                    LOGGER.info(str(check_output(["ls", "-lh", filepath])))
                    self.experiment_tracker.log_artifact(
                        local_path=filepath, artifact_path=artifacts_dir
                    )
