# Adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import logging
import os
from pathlib import Path

from cvlization.dataset.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.specs.prediction_tasks import InstanceSegmentation
from cvlization.torch.torch_training_pipeline import TorchTrainingPipeline
from cvlization.torch.net.instance_segmentation.torchvision import (
    TorchvisionInstanceSegmentationModelFactory,
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
        dataset_builder = self.create_dataset()
        self.num_classes = num_classes = dataset_builder.num_classes
        model = self.create_model()
        training_pipeline = self.create_training_pipeline(model)
        training_pipeline.fit(dataset_builder)

    def create_model(self):
        # Use TorchvisionDetectionModelFactory.list_models() to get a list of available models.
        model = TorchvisionInstanceSegmentationModelFactory(
            # TODO: check num_classes against the dataset
            num_classes=self.num_classes,
            net=self.args.net,
            lightning=True,
            lr=0.0001,
            pretrained=True,
        ).run()
        assert model is not None, "Model is None."
        return model

    def create_dataset(self):
        # Use TorchvisionDatasetBuilder.list_datasets() to get a list of available datasets.
        cache_dir = str(get_cache_dir())
        LOGGER.info(f"Using cache directory: {cache_dir}")
        dataset_builder = PennFudanPedestrianDatasetBuilder(
            flavor="torchvision", include_masks=True, label_offset=1, data_dir=cache_dir
        )
        return dataset_builder

    def create_training_pipeline(self, model):
        assert model is not None, "Model is None."
        training_pipeline = TorchTrainingPipeline(
            model=model,
            # Data loader parameters.
            collate_method="zip",
            train_batch_size=4,
            val_batch_size=2,
            # Training loop parameters.
            epochs=10,
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
    python -m examples.instance_segmentation.torchvision.train
    """

    from argparse import ArgumentParser

    options = TorchvisionInstanceSegmentationModelFactory.model_names()
    parser = ArgumentParser(
        epilog=f"""
            Options for net: {options} ({len(options)} of them).
            
            """
    )
    parser.add_argument("--net", type=str, default="maskrcnn_resnet50_fpn")
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()
    TrainingSession(args).run()
