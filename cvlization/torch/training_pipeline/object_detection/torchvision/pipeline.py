from dataclasses import dataclass
import logging
from typing import Union, Callable

from cvlization.specs.ml_framework import MLFramework
from cvlization.specs.prediction_tasks import ObjectDetection
from cvlization.training_pipeline import TrainingPipeline
from cvlization.torch.training_pipeline.object_detection.torchvision.model import (
    TorchvisionDetectionModelFactory,
)
from cvlization.lab.experiment import Experiment


LOGGER = logging.getLogger(__name__)

@dataclass
class TorchvisionObjectDetection:
    # Model arch and task.
    net: str = "fcos_resnet50_fpn"
    pretrained: bool = True

    loss_function_included_in_model: bool = True

    # Precision
    precision: str = "fp32"  # "fp16", "fp32"

    # Data
    num_workers: int = 0
    collate_method: Union[str, Callable] = "zip"  # "zip", None

    # Optimizer
    lr: float = 0.0001
    optimizer_name: str = "Adam"
    optimizer_kwargs: dict = None
    lr_scheduler_name: str = None
    lr_scheduler_kwargs: dict = None
    n_gradients: int = 1  # for gradient accumulation
    epochs: int = 10
    train_batch_size: int = 8
    train_steps_per_epoch: int = None
    val_batch_size: int = 1
    val_steps_per_epoch: int = None
    check_val_every_n_epoch: int = 5
    reduce_lr_patience: int = 5
    early_stop_patience: int = 10

    # Framework
    lightning: bool = True

    # Logging
    experiment_tracker: str = None
    experiment_name: str = "torchvision_object_detection"
    run_name: str = None

    # Debugging
    data_only: bool = False  # No training, only run through data.
    

    def train(self, dataset_builder):
        num_classes = dataset_builder.num_classes
        model = TorchvisionDetectionModelFactory(
            num_classes=num_classes,
            net=self.net,
            lightning=self.lightning,
            lr=self.lr,  # TODO: lr is specified in 2 places
            pretrained=self.pretrained,
        ).run()
        # TODO: the current TrainingPipeline will be refactored.
        training_pipeline = TrainingPipeline(
            # Annotating the ml framework helps the training pipeline to use
            #   appropriate adapters for the dataset.
            ml_framework=MLFramework.PYTORCH,
            model=model,
            # Data loader parameters.
            collate_method=self.collate_method,
            train_batch_size=self.train_batch_size,
            val_batch_size=self.val_batch_size,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            # Training loop parameters.
            epochs=self.epochs,
            train_steps_per_epoch=self.train_steps_per_epoch,
            # Optimizer parameters.
            optimizer_name=self.optimizer_name,
            lr=self.lr,
            n_gradients=self.n_gradients,
            # Experiment tracking/logging.
            experiment_tracker=self.experiment_tracker,
            # Misc parameters.
            loss_function_included_in_model=self.loss_function_included_in_model,
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
