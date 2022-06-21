import logging

from cvlization.specs.ml_framework import MLFramework
from cvlization.torch.data.torchvision_dataset_builder import TorchvisionDatasetBuilder
from cvlization.lab.kitti_tiny import KittiTinyDatasetBuilder
from cvlization.lab.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.specs.prediction_tasks import ObjectDetection
from cvlization.training_pipeline import TrainingPipeline
from cvlization.lab.experiment import Experiment
from cvlization.torch.net.object_detection.torchvision import (
    TorchvisionDetectionModelFactory,
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
            prediction_task=ObjectDetection(n_categories=num_classes),
            # Dataset and transforms.
            dataset_builder=dataset_builder,
            # Training pipeline: model, trainer, optimizer.
            training_pipeline=training_pipeline,
        ).run()

    def create_model(self):
        # Use TorchvisionDetectionModelFactory.list_models() to get a list of available models.
        model = TorchvisionDetectionModelFactory(
            num_classes=self.num_classes,
            net=self.args.net,
            lightning=True,
            lr=0.0001,  # TODO: lr is specified in 2 places
        ).run()
        return model

    def create_dataset(self):
        LOGGER.info(
            f"Available dataset builders: {KittiTinyDatasetBuilder(), PennFudanPedestrianDatasetBuilder(), TorchvisionDatasetBuilder.list_dataset_builders()}"
        )
        dataset_builder = KittiTinyDatasetBuilder(flavor="torchvision", label_offset=1)
        # dataset_builder = PennFudanPedestrianDatasetBuilder(
        #     flavor="torchvision", include_masks=False, label_offset=1
        # )
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
    python -m examples.object_detection.torchvision.train
    """

    from argparse import ArgumentParser

    options = TorchvisionDetectionModelFactory.model_names()
    parser = ArgumentParser(
        epilog=f"""
            Options for net: {options} ({len(options)} of them). Though you probably want a fpn in it.
            """
    )
    parser.add_argument("--net", type=str, default="fcos_resnet50_fpn")
    # Alternative options:
    # net="retinanet_resnet50_fpn",
    # net="fasterrcnn_resnet50_fpn", and more
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()
    TrainingSession(args).run()
