from dataclasses import dataclass
import tensorflow as tf
import tensorflow_datasets as tfds

from ..transforms.image_augmentation import normalize
from ...data.dataset_builder import BaseDatasetBuilder, Dataset, DatasetProvider
from ...specs import (
    ImageAugmentationSpec,
)
from ...transforms.image_augmentation_builder import ImageAugmentationBuilder


@dataclass
class TFDatasetBuilder(BaseDatasetBuilder):
    dataset_name: str
    as_numpy: bool = False
    # image augmentation for training data
    train_image_augmentation: ImageAugmentationSpec = None
    val_image_augmentation: ImageAugmentationSpec = None
    # image normalization
    image_mean: tuple = None
    image_std: tuple = None

    @property
    def dataset_provider(self):
        return DatasetProvider.TENSORFLOW_DATASETS

    def training_dataset(self):
        """
        Returns:
            A tf.data.Dataset object without batching applied."""
        train_ds = tfds.load(
            self.dataset_name,
            split="train",
            as_supervised=True,
            batch_size=None,
            shuffle_files=True,
        )  # .prefetch(tf.data.experimental.AUTOTUNE)

        train_ds = self.apply_augmentation(train_ds, self.train_image_augmentation)

        # if self.as_numpy:
        #     # TODO: as_numpy does not return a familiar data structure.
        #     # Consider removing this option.
        #     train_ds = tfds.as_numpy(train_ds)
        return train_ds

    def validation_dataset(self):
        ds = tfds.load(
            self.dataset_name,
            split="test",
            as_supervised=True,
            batch_size=None,
        )
        ds = self.apply_augmentation(ds, self.val_image_augmentation)
        return ds

    def apply_augmentation(
        self, dataset: Dataset, image_augmentation: ImageAugmentationSpec
    ) -> Dataset:
        """
        args:
            dataset (torch.utils.data.Dataset)): dataset to apply augmentation to

        returns:
            augmented_dataset (torch.utils.data.Dataset): dataset with augmentation applied

        """
        augment = ImageAugmentationBuilder(spec=image_augmentation).run()
        dataset = dataset.map(
            normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset
