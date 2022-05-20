import logging
import torch
from pytorch_lightning.core import LightningModule
from torchmetrics.detection.map import MeanAveragePrecision

from cvlization.specs.ml_framework import MLFramework
from cvlization.lab.kitti_tiny import KittiTinyDatasetBuilder
from cvlization.specs.prediction_tasks import ObjectDetection
from cvlization.training_pipeline import TrainingPipeline
from cvlization.lab.experiment import Experiment
from cvlization.torch.net.object_detection.torchvision import (
    TorchvisionDetectionModelFactory,
)


LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    """
    python -m examples.object_detection.torchvision.train
    """

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()

    # Use TorchvisionDetectionModelFactory.list_models() to get a list of available models.
    model = TorchvisionDetectionModelFactory(
        num_classes=3,
        net="fcos_resnet50_fpn",
        lightning=True,
        lr=0.0001,  # TODO: lr is specified in 2 places
    ).run()

    training_pipeline = TrainingPipeline(
        # Annotating the ml framework helps the training pipeline to use
        #   appropriate adapters for the dataset.
        ml_framework=MLFramework.PYTORCH,
        model=model,
        # Data loader parameters.
        collate_method="zip",
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

    Experiment(
        # The interface (inputs and outputs) of the model.
        prediction_task=ObjectDetection(n_categories=3),
        # Dataset and transforms.
        dataset_builder=KittiTinyDatasetBuilder(),
        # Training pipeline: model, trainer, optimizer.
        training_pipeline=training_pipeline,
    ).run()
