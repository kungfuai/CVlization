import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from cvlization.lab.model_specs import ImageClassification
from ..data.splitted_dataset import SplittedDataset
from ..specs import ModelInput, ModelTarget, ModelSpec as PredictionTask, DataColumnType
from ..keras.transforms.image_transforms import normalize, ImageAugmentation

# TODO: explore https://cgarciae.github.io/dataget/


def datasets() -> Dict[str, SplittedDataset]:
    return {
        ds.dataset_key: ds
        for ds in [
            TFDSImageDataset(
                dataset_key="cifar10_tfds",
                shuffle_size=20000,
                image_augmentation=ImageAugmentation(
                    image_height=32,
                    image_width=32,
                    flip_left_right=True,
                    random_brightness=0,  # 0.3,
                    random_contrast=0,  # 1.2,
                    random_saturation=None,
                    random_hue=None,
                    random_rotation=None,
                    random_crop=True,
                    scale_before_crop=1.25,
                ),
            ),
            TorchVisionDataset(dataset_key="cifar10_torchvision"),
        ]
    }


@dataclass
class TFDSImageDataset(SplittedDataset):
    dataset_key: str = "cifar10_tfds"
    as_numpy: bool = False
    image_augmentation: ImageAugmentation = None
    shuffle_size: int = 3072

    def __post_init__(self):
        self._tfds_dataset_name = self.dataset_key.replace("_tfds", "")

    def supported_prediction_tasks(self) -> List[PredictionTask]:
        if self.dataset_key.startswith("cifar10"):
            # `num_classes` need to be known before a neural net can be constructed.
            # image height and width, on the other hand, can be specified after the model
            #   is constructed. The exact dimension of the network is fixed once one batch
            #   of data passes through the network.
            return [ImageClassification(num_classes=10)]
        elif self.dataset_key.startswith("mnist"):
            return [ImageClassification(num_classes=10)]
        elif self.dataset_key.startswith("cifar100"):
            return [ImageClassification(num_classes=100)]
        else:
            raise NotImplementedError

    def prepare_augmentation_tf(self):
        import tensorflow as tf
        from tensorflow.keras import layers

        steps = []
        if self.image_augmentation.flip_left_right:
            steps.append(layers.RandomFlip("horizontal_and_vertical"))
        if self.image_augmentation.random_rotation:
            steps.append(layers.RandomRotation(self.image_augmentation.random_rotation))
        if self.image_augmentation.random_crop:
            assert self.image_augmentation.scale_before_crop > 1
            factor = self.image_augmentation.scale_before_crop - 1
            steps.extend(
                [
                    layers.RandomHeight((0, factor)),
                    layers.RandomWidth((0, factor)),
                    layers.RandomCrop(
                        height=self.image_augmentation.image_height,
                        width=self.image_augmentation.image_width,
                    ),
                ]
            )
        if (self.image_augmentation.random_contrast or 0) > 1:
            steps.append(
                layers.RandomContrast(self.image_augmentation.random_contrast - 1)
            )
        # TODO: add additional augmentations

        augment_fn = tf.keras.Sequential(steps)

        def augment(image, label):
            image = augment_fn(image)
            return image, label

        self.augment = augment
        return augment

    def training_dataset_as_rich_dataframe(self):
        """
        From generic to specific:
        RichDataFrame -> MLDataset -> DataLoader / tf.data.Dataset
        """
        raise NotImplementedError

    def training_dataset(self, batch_size: int = None):
        import tensorflow_datasets as tfds

        train_ds = tfds.load(
            self._tfds_dataset_name,
            split="train",
            as_supervised=True,
            batch_size=batch_size,
            shuffle_files=True,
        )  # .prefetch(tf.data.experimental.AUTOTUNE)
        # TODO: as_numpy does not return a familiar data structure.
        if self.as_numpy:
            train_ds = tfds.as_numpy(train_ds)
        return train_ds

    def validation_dataset(self, batch_size: int = None):
        import tensorflow_datasets as tfds

        val_ds = tfds.load(
            self._tfds_dataset_name,
            split="test",
            as_supervised=True,
            batch_size=batch_size,
        )
        if self.as_numpy:
            val_ds = tfds.as_numpy(val_ds)
        return val_ds

    def transform_training_dataset_tf(self, training_dataset):
        import tensorflow as tf

        augment = self.prepare_augmentation_tf()
        if self.shuffle_size:
            training_dataset = training_dataset.shuffle(self.shuffle_size)
        training_dataset = training_dataset.map(
            augment, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        training_dataset = training_dataset.map(
            normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return training_dataset

    def transform_validation_dataset_tf(self, validation_dataset):
        import tensorflow as tf

        validation_dataset = validation_dataset.map(
            normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return validation_dataset

    def transform_training_dataset_torch(self, training_dataset):
        import torch
        import torchvision.transforms as transforms

        # TODO: allow using imgaug as a default
        # TODO: transforms should be in a separate class
        # https://github.com/kungfuai/mtrx_2/blob/1b5ff963f4b732883e95e1f86dfbecbb95a7a9ff/src/data/transforms.py#L31
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        class IterableImageDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                for image, label in training_dataset:
                    image = image.numpy() / 255
                    label = label.numpy()
                    image = torch.cat(
                        [
                            torch.unsqueeze(transform_train(image[i]), 0)
                            for i in range(len(image))
                        ]
                    )
                    yield image, label

        return IterableImageDataset()

    def transform_validation_dataset_torch(self, validation_dataset):
        import torch
        import torchvision.transforms as transforms

        transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        class IterableImageDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                for image, label in validation_dataset:
                    image = image.numpy() / 255
                    label = label.numpy()
                    image = torch.cat(
                        [
                            torch.unsqueeze(transform_val(image[i]), 0)
                            for i in range(len(image))
                        ]
                    )
                    # print(image.shape, type(image))
                    # raise NotImplementedError("222")

                    yield image, label

        return IterableImageDataset()


@dataclass
class TorchVisionDataset(SplittedDataset):
    dataset_key: str = "cifar10_torchvision"
    dataset_type = None  # TODO: deal with obj det etc
    data_dir: str = "./data"

    def __post_init__(self):
        self.model_inputs = [
            ModelInput(
                key="image",
                column_type=DataColumnType.IMAGE,
            )
        ]
        self.model_targets = [
            ModelTarget(
                key="label",
                column_type=DataColumnType.NUMERICAL,
            )
        ]

    def get_transform_for_training_data(self):
        """Returns transform and target_transform."""
        import torchvision.transforms as transforms

        if self.dataset_key.startswith("cifar"):
            return (
                transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                ),
                None,
            )
        else:
            return (
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                None,
            )

    def get_transform_for_validation_data(self):
        """Returns transform and target_transform."""
        import torchvision.transforms as transforms

        if self.dataset_key.startswith("cifar"):
            return (
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                ),
                None,
            )
        else:
            return (
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                None,
            )

    def get_dataset_classname(self):
        dataset_classname = self.dataset_key.replace("_torchvision", "").upper()
        return dataset_classname

    def training_dataset(self, batch_size: int = 32):
        # TODO: consider moving DataLoader out of this function, and keep the dataset
        #    interface more generic.
        import torch
        import torchvision

        transform, target_transform = self.get_transform_for_training_data()
        dataset_classname = self.get_dataset_classname()
        if hasattr(torchvision.datasets, dataset_classname):
            dataset_class = getattr(torchvision.datasets, dataset_classname)
            train_data = dataset_class(
                root=self.data_dir,
                train=True,
                download=True,
                transform=transform,
                target_transform=target_transform,
            )
        else:
            raise ValueError(f"Unknown torchvision dataset {dataset_classname}")

        train_data_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=8
        )
        return train_data_loader

    def validation_dataset(self, batch_size: int = 32):
        import torch
        import torchvision

        transform, target_transform = self.get_transform_for_validation_data()
        dataset_classname = self.get_dataset_classname()
        if hasattr(torchvision.datasets, dataset_classname):
            dataset_class = getattr(torchvision.datasets, dataset_classname)
            val_data = dataset_class(
                root="./data",
                train=False,
                download=True,
                transform=transform,
                target_transform=target_transform,
            )
        else:
            raise ValueError(f"Unknown torchvision dataset {dataset_classname}")
        val_data_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=8
        )
        return val_data_loader

    def _transform_to_tf_dataset(self, dataset):
        import tensorflow as tf

        def gen():
            for batch in dataset:
                inputs = batch[0]
                targets = batch[1]
                yield (
                    tuple([tf.convert_to_tensor(inputs.numpy().transpose(0, 2, 3, 1))]),
                    tuple(
                        [
                            tf.convert_to_tensor(targets.numpy().astype(np.float32)),
                        ]
                    ),
                )

        output_signature = (
            tuple(
                [tf.TensorSpec(shape=None, dtype=tf.float32)] * len(self.model_inputs)
            ),
            tuple(
                [tf.TensorSpec(shape=None, dtype=tf.float32)] * len(self.model_targets)
            ),
        )

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        return ds

    def transform_training_dataset_tf(self, training_dataset):
        ds = self._transform_to_tf_dataset(training_dataset)
        return ds

    def transform_validation_dataset_tf(self, validation_dataset):
        ds = self._transform_to_tf_dataset(validation_dataset)
        return ds