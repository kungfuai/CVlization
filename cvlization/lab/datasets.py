import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from cvlization.lab.model_specs import ImageClassification
from ..data.splitted_dataset import SplittedDataset
from .dataset_utils import torchvision_dataset_classnames
from ..specs import (
    ModelInput,
    ModelTarget,
    ModelSpec as PredictionTask,
    DataColumnType,
    ImageAugmentation,
    ImageAugmentationProvider,
)
from ..keras.transforms.image_transforms import (
    normalize,
    ImageAugmentation as KerasImageAugmentation,
)
from ..transforms.image_augmentation_builder import ImageAugmentationBuilder
from . import dataset_adaptors

# TODO: explore https://cgarciae.github.io/dataget/


def datasets() -> Dict[str, SplittedDataset]:
    example_datasets = {
        ds.dataset_key: ds
        for ds in [
            TFDSImageDataset(
                dataset_key="cifar10_tfds",
                shuffle_size=20000,
                image_augmentation=KerasImageAugmentation(
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
            TorchVisionDataset(
                dataset_key="cifar10_torchvision",
                image_augmentation=ImageAugmentation(
                    provider=ImageAugmentationProvider.TORCHVISION,
                    config={
                        "transformers": [
                            {
                                "type": "RandomCrop",
                                "kwargs": {"size": 32, "padding": 4},
                            },
                            {"type": "RandomHorizontalFlip"},
                            {"type": "ToTensor"},
                            {
                                "type": "Normalize",
                                "kwargs": {
                                    "mean": (0.4914, 0.4822, 0.4465),
                                    "std": (0.2023, 0.1994, 0.2010),
                                },
                            },
                        ]
                    },
                ),
            ),
        ]
    }
    for torchvision_dataset_classname in torchvision_dataset_classnames():
        dataset_key = f"{torchvision_dataset_classname.lower()}_torchvision"
        if dataset_key in example_datasets:
            continue
        example_datasets[dataset_key] = TorchVisionDataset(
            dataset_key=f"{torchvision_dataset_classname}_torchvision",
        )
    return example_datasets


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
    # image augmentation for training data
    image_augmentation: ImageAugmentation = None
    image_mean: tuple = None
    image_std: tuple = None

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
        val_augmentation_steps = [{"type": "ToTensor"}]
        image_mean, image_std = self.image_mean, self.image_std
        if image_mean is None or image_std is None:
            image_mean, image_std = self.get_image_dataset_mean_std()
        if image_mean is not None and image_std is not None:
            val_augmentation_steps.append(
                {
                    "type": "Normalize",
                    "kwargs": {"mean": image_mean, "std": image_std},
                },
            )
        self.val_image_augmentation = ImageAugmentation(
            provider=ImageAugmentationProvider.TORCHVISION,
            config={
                "transformers": val_augmentation_steps,
            },
        )
        if self.image_augmentation is None:
            self.image_augmentation = self.val_image_augmentation

    def get_image_dataset_mean_std(self):
        if self.dataset_key.startswith("cifar"):
            return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        else:
            return None, None

    def get_transform_for_training_data(self):
        target_transform = None
        if self.dataset_key.lower().startswith("vocdetection"):
            target_transform = dataset_adaptors.VOCDetectionAdapter.target_transform
        return (
            ImageAugmentationBuilder(spec=self.image_augmentation).run(),
            target_transform,
        )

    def get_transform_for_validation_data(self):
        target_transform = None
        if self.dataset_key.lower().startswith("vocdetection"):
            target_transform = dataset_adaptors.VOCDetectionAdapter.target_transform
        elif self.dataset_key.lower().startswith("vocsegmentation"):
            target_transform = dataset_adaptors.VOCSegmentationAdapter.target_transform
        return (
            ImageAugmentationBuilder(spec=self.val_image_augmentation).run(),
            target_transform,
        )

    def apply_augmentation(self, dataset):
        """
        args:
            dataset (torch.utils.data.Dataset)): dataset to apply augmentation to

        returns:
            augmented_dataset (torch.utils.data.Dataset): dataset with augmentation applied

        """
        import torch

        augment = ImageAugmentationBuilder(spec=self.image_augmentation).run()

        class AugmentedDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.dataset = dataset

            def __getitem__(self, index):
                example = self.dataset[index]
                augmented_example = augment(example)
                return augmented_example

            def __len__(self):
                return len(self.dataset)

        return AugmentedDataset()

    def get_dataset_classname(self):
        dataset_classname_lowercase = self.dataset_key.replace(
            "_torchvision", ""
        ).lower()
        for dataset_classname in torchvision_dataset_classnames():
            if dataset_classname.lower() == dataset_classname_lowercase:
                return dataset_classname
        raise ValueError(
            f"Cannot find dataset in torchvision: {dataset_classname_lowercase} (case insensitive)"
        )

    def training_dataset(self, batch_size: int = 32):
        # TODO: consider moving DataLoader out of this function, and keep the dataset
        #    interface more generic.
        import torch
        import torchvision

        transform, target_transform = self.get_transform_for_training_data()
        val_transform, _ = self.get_transform_for_validation_data()
        dataset_classname = self.get_dataset_classname()
        should_augment_image_and_target_together = False
        if self.dataset_key.lower().startswith("voc"):
            if (
                self.image_augmentation.provider
                != ImageAugmentationProvider.TORCHVISION
            ):
                # torchvision.transforms apply on images but not together with targets.
                should_augment_image_and_target_together = True
        if hasattr(torchvision.datasets, dataset_classname):
            dataset_class = getattr(torchvision.datasets, dataset_classname)
            try:
                train_data = dataset_class(
                    root=self.data_dir,
                    train=True,
                    download=True,
                    transform=transform,
                    target_transform=target_transform,
                )
            except:
                train_data = dataset_class(
                    root=self.data_dir,
                    image_set="train",  # applys to VOC datasets
                    download=True,
                    # VOC datasets should use val_transform for training data. Augmentation is applied separately.
                    transform=val_transform,
                    target_transform=target_transform,
                )
        else:
            raise ValueError(f"Unknown torchvision dataset {dataset_classname}")

        if should_augment_image_and_target_together:
            assert False
            train_data = self.apply_augmentation(train_data)

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
            try:
                val_data = dataset_class(
                    root="./data",
                    train=False,
                    download=True,
                    transform=transform,
                    target_transform=target_transform,
                )
            except:
                val_data = dataset_class(
                    root=self.data_dir,
                    image_set="val",
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
