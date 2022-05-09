# TODO: move to cvlization/torch
import torch
from dataclasses import dataclass

from ..data.dataset_builder import DatasetBuilder
from ..specs import (
    ModelInput,
    ModelTarget,
    ModelSpec as PredictionTask,
    DataColumnType,
    ImageAugmentationSpec,
    ImageAugmentationProvider,
)
from . import dataset_adaptors
from .dataset_utils import torchvision_dataset_classnames
from ..transforms.image_augmentation_builder import ImageAugmentationBuilder


@dataclass
class TorchvisionDatasetBuilder(DatasetBuilder):
    """
    A dataset builder for torchvision datasets.
    """

    dataset_key: str = "cifar10"
    dataset_type = None  # TODO: deal with obj det etc
    data_dir: str = "./data"
    # image augmentation for training data
    image_augmentation: ImageAugmentationSpec = None
    image_mean: tuple = None
    image_std: tuple = None

    def run(self):
        pass

    def __post_init__(self):
        self._infer_prediction_task()
        self._configure_augmentation()

    def _infer_prediction_task(self):
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

    def _configure_augmentation(self):
        # TODO: this is assuming torchvision transforms
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
        self.val_image_augmentation = ImageAugmentationSpec(
            provider=ImageAugmentationProvider.TORCHVISION,
            config={
                "transformers": val_augmentation_steps,
            },
        )
        if self.image_augmentation is None:
            self.image_augmentation = self.val_image_augmentation

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
