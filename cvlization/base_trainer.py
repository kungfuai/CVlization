import logging
import os
import random
from skimage.io import imsave
import numpy as np

from .data.ml_dataset import MLDataset
from .specs import DataColumnType


LOGGER = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(
        self,
        train_dataset: MLDataset,
        val_dataset: MLDataset = None,
        experiment_tracker=None,
        log_input_images: bool = False,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.experiment_tracker = experiment_tracker
        self.log_input_images = log_input_images

    def run(self):
        return self.train()

    def train(self):
        if self.epochs <= 0:
            return
        self._train_val_datasets_should_have_matching_inputs_and_targets()
        self._training_data_stats()
        if self.experiment_tracker is not None and self.log_input_images:
            self._log_input_images(self.train_dataset, artifact_dir="batches_train")
            self._log_input_images(self.val_dataset, artifact_dir="batches_val")
        elif self.log_input_images:
            LOGGER.warning(
                "Experiment tracker is not set. No input images will be logged."
            )
        self._training_loop()

    def _training_loop(self):
        raise NotImplementedError

    def _train_val_datasets_should_have_matching_inputs_and_targets(self):
        if isinstance(self.train_dataset, MLDataset) and isinstance(
            self.val_dataset, MLDataset
        ):
            assert len(self.train_dataset.model_inputs) == len(
                self.val_dataset.model_inputs
            )
            assert len(self.train_dataset.model_targets) == len(
                self.val_dataset.model_targets
            )

    def _find_best_lr(self, train_data: MLDataset, val_data: MLDataset) -> float:
        raise NotImplementedError

    # TODO: _save_image can be part of ExperimentTracker.
    def _save_image(self, im: np.array, filepath: str):
        if hasattr(im, "numpy"):
            im = im.numpy()
        im = im.astype(np.float32)
        im = (im - im.min()) / (im.max() - im.min())
        im = (im * 255).astype(np.uint8)
        # TODO: use PIL instead.
        imsave(filepath, im)

    def _log_input_images(
        self,
        ds: MLDataset,
        num_batches: int = 10,
        artifact_dir: str = "batches_train",
    ):
        if isinstance(ds, MLDataset):
            self._log_input_images_with_ml_dataset(ds, num_batches, artifact_dir)
        else:
            raise ValueError(
                "To log input images, the dataset must be either a MLDataset or a tf.data.Dataset"
            )

    def _log_input_images_with_ml_dataset(
        self, ds: MLDataset, num_batches: int = 10, artifacts_dir: str = "batches_train"
    ):
        total_examples = len(ds)
        indices = list(range(total_examples))
        sampled_indices = random.choices(indices, k=num_batches)
        first_model_target = ds.model_targets[0]
        if first_model_target.column_type == DataColumnType.BOOLEAN or (
            first_model_target.column_type == DataColumnType.CATEGORICAL
            and first_model_target.n_categories == 2
        ):
            should_add_target_value_to_filename = True
        else:
            should_add_target_value_to_filename = False
        for i in sampled_indices:
            inputs, targets, _ = ds[i]
            first_target_tensor = targets[0]
            for model_input, input_tensor in zip(ds.model_inputs, inputs):
                if model_input.column_type == DataColumnType.IMAGE:
                    if should_add_target_value_to_filename:
                        target_value = first_target_tensor
                        if isinstance(target_value, list) or isinstance(
                            target_value, np.ndarray
                        ):
                            target_value = target_value[-1]
                        # TODO: hard coded for now
                        filepath = f"/tmp/{self.name}/label{target_value}_example{i}_{model_input.key}.png"
                    else:
                        filepath = f"/tmp/{self.name}/example{i}_{model_input.key}.png"
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    self._save_image(input_tensor, filepath)
                    self.experiment_tracker.log_artifact(
                        local_path=filepath, artifact_path=artifacts_dir
                    )

    def _training_data_stats(self):
        if isinstance(self.train_dataset, MLDataset):
            self._training_data_stats_with_ml_dataset()
        else:
            LOGGER.warning(
                f"Sample data stats for {type(self.train_dataset)} is not implemented yet."
            )

    def _training_data_stats_with_ml_dataset(self):
        # TODO: this is using keras sequence for now. Need a more generic implementation.
        from .keras import sequence

        assert isinstance(self.train_dataset, MLDataset)
        train_seq = sequence.from_ml_dataset(self.train_dataset)
        for j, (inputs, targets, sample_weight) in enumerate(train_seq):
            LOGGER.info(f"batch {j}: {len(inputs[0])} examples")
            for i, input_array in enumerate(inputs):
                LOGGER.info(
                    f"{self.train_dataset.model_inputs[i].key}: mean={input_array.mean()}, shape={input_array.shape}"
                )
            for i, target_array in enumerate(targets):
                LOGGER.info(
                    f"{self.train_dataset.model_targets[i].key}: mean={target_array.mean()}, shape={input_array.shape}"
                )
            LOGGER.info(f"sample weights: avg = {sample_weight.mean()}")
            if j >= 1:
                break
