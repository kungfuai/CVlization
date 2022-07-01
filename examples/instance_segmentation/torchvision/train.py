# Adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import logging

from cvlization.specs.ml_framework import MLFramework
from cvlization.lab.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.specs.prediction_tasks import InstanceSegmentation
from cvlization.training_pipeline import TrainingPipeline
from cvlization.lab.experiment import Experiment
from cvlization.torch.net.instance_segmentation.torchvision import (
    TorchvisionInstanceSegmentationModelFactory,
)


LOGGER = logging.getLogger(__name__)


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        dataset_builder = self.create_dataset()
        self.num_classes = num_classes = dataset_builder.num_classes
        model = self.create_model()
        training_pipeline = self.create_training_pipeline(model)
        Experiment(
            # The interface (inputs and outputs) of the model.
            prediction_task=InstanceSegmentation(n_categories=num_classes),
            # Dataset and transforms.
            dataset_builder=dataset_builder,
            # Training pipeline: model, trainer, optimizer.
            training_pipeline=training_pipeline,
        ).run()

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
        return model

    def create_dataset(self):
        # Use TorchvisionDatasetBuilder.list_datasets() to get a list of available datasets.
        dataset_builder = PennFudanPedestrianDatasetBuilder(
            flavor="torchvision", include_masks=True, label_offset=1
        )
        return dataset_builder

    def create_training_pipeline(self, model):
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
