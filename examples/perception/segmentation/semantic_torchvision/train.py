"""
Adapted from https://github.com/pytorch/vision/blob/main/references/segmentation/train.py
with the help of `pytorch-lightning`.
"""
import logging
import os
from pathlib import Path

from cvlization.specs.ml_framework import MLFramework
from cvlization.specs import ImageAugmentationSpec, ImageAugmentationProvider
from cvlization.data.dataset_builder import TransformedMapStyleDataset
from cvlization.transforms.image_augmentation_builder import ImageAugmentationBuilder
from cvlization.transforms.example_transform import ExampleTransform
from cvlization.specs.prediction_tasks.semantic_segmentation import SemanticSegmentation
from cvlization.legacy_training_pipeline import LegacyTrainingPipeline
from cvlization.dataset.stanford_background import StanfordBackgroundDatasetBuilder
from cvlization.torch.net.semantic_segmentation.torchvision import (
    TorchvisionSemanticSegmentationModelFactory,
)


LOGGER = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Get the CVlization cache directory for datasets."""
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = Path(cache_home) / "cvlization" / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        self.prediction_task = SemanticSegmentation()
        dataset_builder = self.create_dataset()
        self.num_classes = self.get_num_classes_from_dataset()
        model = self.create_model()
        training_pipeline = self.create_training_pipeline(model)
        # Experiment(
        #     # The interface (inputs and outputs) of the model.
        #     prediction_task=self.prediction_task,
        #     # Dataset and transforms.
        #     dataset_builder=dataset_builder,
        #     # Training pipeline: model, trainer, optimizer.
        #     training_pipeline=training_pipeline,
        # ).run()
        training_pipeline.create_dataloaders(dataset_builder).create_trainer().run()

    def create_model(self):
        model = TorchvisionSemanticSegmentationModelFactory(
            num_classes=self.num_classes,
            net=self.args.net,
            lightning=True,
            pretrained_backbone=True,
            lr=0.0001,  # TODO: lr is specified in 2 places
        ).run()
        return model

    def get_num_classes_from_dataset(self):
        cache_dir = str(get_cache_dir())
        dataset_builder = StanfordBackgroundDatasetBuilder(
            flavor=None, label_offset=1, data_dir=cache_dir
        )
        return len(dataset_builder.CLASSES)

    def create_dataset(self):
        cache_dir = str(get_cache_dir())
        LOGGER.info(f"Available dataset builders: {StanfordBackgroundDatasetBuilder()}")
        LOGGER.info(f"Using cache directory: {cache_dir}")
        dataset_builder = StanfordBackgroundDatasetBuilder(
            flavor=None, label_offset=1, data_dir=cache_dir
        )
        aug_spec = ImageAugmentationSpec(
            provider=ImageAugmentationProvider.ALBUMENTATIONS,
            config={
                "deterministic": True,
                "norm": False,
                "cv_task": "semseg",
                "steps": [
                    {
                        "type": "resize",
                        "probability": 1,
                        "kwargs": {"size": {"height": 300, "width": 300}},
                    },
                ],
            },
        )
        image_augmentation = ImageAugmentationBuilder(spec=aug_spec).run()
        example_transform = ExampleTransform(
            image_augmentation=image_augmentation,
            model_inputs=self.prediction_task.get_model_inputs(),
            model_targets=self.prediction_task.get_model_targets(),
        )

        class DatasetBuilderWithTransform:
            def __init__(self, base_dataset_builder):
                self.base_dataset_builder = base_dataset_builder

            def training_dataset(self):
                ds = self.base_dataset_builder.training_dataset()
                ds = TransformedMapStyleDataset(
                    ds, transform=example_transform.transform_example
                )
                ds = TransformedMapStyleDataset(
                    ds, transform=self.base_dataset_builder.to_torchvision
                )
                return ds

            def validation_dataset(self):
                # Validation data is using the same transforms as training, for now.
                ds = self.base_dataset_builder.validation_dataset()
                ds = TransformedMapStyleDataset(
                    ds, transform=example_transform.transform_example
                )
                ds = TransformedMapStyleDataset(
                    ds, transform=self.base_dataset_builder.to_torchvision
                )
                return ds

        return DatasetBuilderWithTransform(dataset_builder)

    def create_training_pipeline(self, model):
        training_pipeline = LegacyTrainingPipeline(
            # Annotating the ml framework helps the training pipeline to use
            #   appropriate adapters for the dataset.
            ml_framework=MLFramework.PYTORCH,
            model=model,
            prediction_task=self.prediction_task,
            # Data loader parameters.
            # collate_method="zip",
            train_batch_size=8,
            val_batch_size=2,
            # Training loop parameters.
            epochs=50,
            train_steps_per_epoch=100,
            # Optimizer parameters.
            optimizer_name="Adam",
            lr=0.0001,
            n_gradients=1,
            # Experiment tracking/logging.
            experiment_tracker=None,
            # Misc parameters.
            loss_function_included_in_model=True,
        )
        return training_pipeline


if __name__ == "__main__":
    """
    python -m examples.semantic_segmentation.torchvision.train
    """

    from argparse import ArgumentParser

    options = TorchvisionSemanticSegmentationModelFactory.model_names()
    parser = ArgumentParser(
        epilog=f"""
            *** Options for net: {options} ({len(options)} of them). Though you probably want a fpn in it.
            
            """
    )
    parser.add_argument("--net", type=str, default="fcn_resnet50")
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()
    TrainingSession(args).run()
