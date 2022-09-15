import logging

from dataclasses import dataclass
from cvlization.specs.ml_framework import MLFramework
from cvlization.specs import ModelSpec
from cvlization.torch.data.torchvision_dataset_builder import TorchvisionDatasetBuilder
from cvlization.specs.prediction_tasks import ImageClassification
from cvlization.training_pipeline import TrainingPipeline
from cvlization.lab.experiment import Experiment
from cvlization.torch.encoder.torch_image_backbone import image_backbone_names


LOGGER = logging.getLogger(__name__)


class ImageClassificationTrainingPipeline:
    net: str = "resnet18"
    num_classes: int = 10
    num_channels: int = 3
    image_height: int = None
    image_width: int = None
    channels_first: bool = True

    loss_function_included_in_model: bool = False

    collate_method = None
    epochs: int = 50
    train_batch_size: int = 128
    val_batch_size: int = 128
    train_steps_per_epoch: int = None
    val_steps_per_epoch: int = None
    optimizer_name: str = "Adam"
    lr: float = 0.0001
    n_gradients: int = 1  # for gradient accumulation
    experiment_tracker = None
    
    def train(self, dataset_builder=TorchvisionDatasetBuilder(dataset_classname="CIFAR10")):
        prediction_task = ImageClassification(
            n_classes=self.num_classes or dataset_builder.num_classes,
            num_channels=self.num_channels or dataset_builder.num_channels,
            image_height=self.image_height,
            image_width=self.image_width,
            channels_first=self.channels_first,
        )

        training_pipeline = TrainingPipeline(
            ml_framework=MLFramework.PYTORCH,
            model=ModelSpec(
                image_backbone=self.net,
                model_inputs=prediction_task.get_model_inputs(),
                model_targets=prediction_task.get_model_targets(),
            ),
            loss_function_included_in_model=self.loss_function_included_in_model,
            collate_method=self.collate_method,
            epochs=self.epochs,
            train_batch_size=self.train_batch_size,
            val_batch_size=self.val_batch_size,
            train_steps_per_epoch=self.train_steps_per_epoch,
            val_steps_per_epoch=self.val_steps_per_epoch,
            optimizer_name=self.optimizer_name,
            lr=self.lr,
            n_gradients=self.n_gradients,
            experiment_tracker=self.experiment_tracker,
        )

        Experiment(
            # The interface (inputs and outputs) of the model.
            prediction_task=ImageClassification(
                n_classes=self.num_classes or dataset_builder.num_classes,
                num_channels=self.num_channels or dataset_builder.num_channels,
                image_height=self.image_height,
                image_width=self.image_width,
                channels_first=self.channels_first,
            ),
            # Dataset and transforms.
            dataset_builder=dataset_builder,
            # Training pipeline: model, trainer, optimizer.
            training_pipeline=training_pipeline,
        ).run()


