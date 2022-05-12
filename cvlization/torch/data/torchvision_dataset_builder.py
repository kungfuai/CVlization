from dataclasses import dataclass
import torchvision
from typing import List, Tuple, Union, Optional
from ...data.dataset_builder import BaseDatasetBuilder, Dataset, DatasetProvider
from ...specs import (
    ImageAugmentationSpec,
    ModelInput,
    ModelTarget,
    DataColumnType,
    ImageAugmentationProvider,
)
from . import dataset_adaptors
from ...data.dataset_builder import TransformedMapStyleDataset
from ...transforms.image_augmentation_builder import ImageAugmentationBuilder


@dataclass
class TorchvisionDatasetBuilder(BaseDatasetBuilder):
    dataset_classname: str
    # dataset_type = None  # TODO: deal with obj det etc
    data_dir: str = "./data"
    # image augmentation for training data
    training_transform: ImageAugmentationSpec = None
    validation_transform: ImageAugmentationSpec = None
    # image normalization
    image_mean: tuple = None
    image_std: tuple = None

    @property
    def dataset_provider(self):
        return DatasetProvider.TORCHVISION

    def __post_init__(self):
        image_mean, image_std = self.image_mean, self.image_std
        if image_mean is None or image_std is None:
            self.image_mean, self.image_std = self.get_default_image_dataset_mean_std()
        self.training_transform = (
            self.training_transform or self.get_default_training_transform()
        )
        self.validation_transform = (
            self.validation_transform or self.get_default_validation_transform()
        )

    def get_default_image_dataset_mean_std(self):
        if self.dataset_classname.lower().startswith("cifar"):
            return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        else:
            return None, None

    def _prepare_model_spec(self):
        # TODO: is it needed in this class?
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

    def get_default_training_transform(self):
        if self.dataset_classname.lower().startswith("voc"):
            return ImageAugmentationSpec(
                provider=ImageAugmentationProvider.IMGAUG,
                config={
                    "deterministic": True,
                    "norm": False,
                    "cv_task": "detection",
                    "steps": [
                        {
                            "type": "crop",
                            "probability": 1,
                            "kwargs": {"percent": [0.0, 0.2], "keep_size": False},
                        },
                        {"type": "flip_lr", "probability": 0.5, "kwargs": {"p": 1}},
                        {"type": "flip_ud", "probability": 0.5, "kwargs": {"p": 1}},
                    ],
                },
            )
        else:
            steps = [{"type": "ToTensor"}]
            if self.image_mean is not None and self.image_std is not None:
                steps.append(
                    {
                        "type": "Normalize",
                        "kwargs": {"mean": self.image_mean, "std": self.image_std},
                    },
                )
            return ImageAugmentationSpec(
                provider=ImageAugmentationProvider.TORCHVISION,
                config=dict(steps=steps),
            )

    def get_default_validation_transform(self):
        steps = [{"type": "ToTensor"}]
        if self.image_mean is not None and self.image_std is not None:
            steps.append(
                {
                    "type": "Normalize",
                    "kwargs": {"mean": self.image_mean, "std": self.image_std},
                },
            )
        return ImageAugmentationSpec(
            provider=ImageAugmentationProvider.TORCHVISION,
            config=dict(steps=steps),
        )

    def prepare_transforms(self):
        val_augmentation_steps = self.prepare_val_augmentation_steps()
        self.val_image_augmentation = ImageAugmentationSpec(
            provider=ImageAugmentationProvider.TORCHVISION,
            config={
                "steps": val_augmentation_steps,
            },
        )
        if self.train_image_augmentation is None:
            self.train_image_augmentation = self.val_image_augmentation
        self.transform, self.target_transform = self.get_transform_for_training_data()
        (
            self.val_transform,
            self.val_target_transform,
        ) = self.get_transform_for_validation_data()
        return self

    def verify_dataset_classname(self):
        dataset_classname_lowercase = self.dataset_classname.replace(
            "_torchvision", ""
        ).lower()
        for dataset_classname in torchvision_dataset_classnames():
            if dataset_classname.lower() == dataset_classname_lowercase:
                return dataset_classname
        raise ValueError(
            f"Cannot find dataset in torchvision: {dataset_classname_lowercase} (case insensitive)"
        )

    def construct_torchvision_dataset(self, dataset_classname, train: bool = True):
        if hasattr(torchvision.datasets, dataset_classname):
            dataset_class = getattr(torchvision.datasets, dataset_classname)
            # Torchvision datasets can be constructed using one of the following two ways:
            try:
                ds = dataset_class(
                    root=self.data_dir,
                    train=train,
                    download=True,
                )
            except:
                target_transform = None
                if self.dataset_classname.lower().startswith("vocdetection"):
                    target_transform = (
                        dataset_adaptors.VOCDetectionAdapter.target_transform
                    )
                ds = dataset_class(
                    root=self.data_dir,
                    image_set="train" if train else "val",  # applys to VOC datasets
                    download=True,
                    # VOC datasets should use val_transform for training data. Augmentation is applied separately.
                    target_transform=target_transform,
                )
        else:
            raise ValueError(f"Unknown torchvision dataset {dataset_classname}")
        return ds

    def training_dataset(self):
        dataset_classname = self.verify_dataset_classname()
        ds = self.construct_torchvision_dataset(dataset_classname, train=True)
        transform_fn = ImageAugmentationBuilder(spec=self.training_transform).run()
        ds = TransformedMapStyleDataset(ds, transform=transform_fn)
        return ds

    def validation_dataset(self):
        dataset_classname = self.verify_dataset_classname()
        ds = self.construct_torchvision_dataset(dataset_classname, train=True)
        transform_fn = ImageAugmentationBuilder(spec=self.training_transform).run()
        ds = TransformedMapStyleDataset(ds, transform=transform_fn)
        return ds


def torchvision_dataset_classnames():
    return [
        "CIFAR10",
        "CIFAR100",
        "Caltech101",
        "Caltech256",
        "CelebA",
        "Cityscapes",
        "CocoCaptions",
        "CocoDetection",
        "EMNIST",
        "FakeData",
        "FashionMNIST",
        "Flickr30k",
        "Flickr8k",
        "HMDB51",
        "INaturalist",
        "ImageNet",
        "KMNIST",
        "Kinetics",
        "Kinetics400",
        "Kitti",
        "LFWPairs",
        "LFWPeople",
        "LSUN",
        "LSUNClass",
        "MNIST",
        "Omniglot",
        "PhotoTour",
        "Places365",
        "QMNIST",
        "SBDataset",
        "SBU",
        "SEMEION",
        "STL10",
        "SVHN",
        "UCF101",
        "USPS",
        "VOCDetection",
        "VOCSegmentation",
        "VisionDataset",
        "WIDERFace",
        "caltech",
        "celeba",
        "cifar",
        "cityscapes",
        "coco",
        "fakedata",
        "flickr",
        "folder",
        "hmdb51",
        "imagenet",
        "inaturalist",
        "kinetics",
        "kitti",
        "lfw",
        "lsun",
        "mnist",
        "omniglot",
        "phototour",
        "places365",
        "sbd",
        "sbu",
        "semeion",
        "stl10",
        "svhn",
        "ucf101",
        "usps",
        "voc",
        "widerface",
    ]
