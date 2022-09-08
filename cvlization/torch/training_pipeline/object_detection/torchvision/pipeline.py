import logging

from cvlization.specs.ml_framework import MLFramework
from cvlization.specs.prediction_tasks import ObjectDetection
from cvlization.training_pipeline import TrainingPipeline
from cvlization.torch.training_pipeline.object_detection.torchvision.model import (
    TorchvisionDetectionModelFactory,
)
from cvlization.lab.experiment import Experiment


class TorchvisionObjectDetection:
    def __init__(self, net: str):
        self._net = net

    def train(self, dataset_builder):
        num_classes = dataset_builder.num_classes
        model = TorchvisionDetectionModelFactory(
            num_classes=num_classes,
            net=self._net,
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
            check_val_every_n_epoch=1,
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
            prediction_task=ObjectDetection(n_categories=num_classes),
            # Dataset and transforms.
            dataset_builder=dataset_builder,
            # Training pipeline: model, trainer, optimizer.
            training_pipeline=training_pipeline,
        ).run()
    
    @classmethod
    def model_names(cls):
        return TorchvisionDetectionModelFactory.model_names()
