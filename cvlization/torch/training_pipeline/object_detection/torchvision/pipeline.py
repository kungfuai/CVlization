from dataclasses import dataclass
import torch
from pytorch_lightning import LightningModule
from typing import Union
import logging
from cvlization.specs.prediction_tasks import ObjectDetection
from cvlization.torch.torch_training_pipeline import TorchTrainingPipeline
from cvlization.torch.training_pipeline.object_detection.torchvision.model import (
    TorchvisionDetectionModelFactory,
)


LOGGER = logging.getLogger(__name__)

@dataclass
class TorchvisionObjectDetection(TorchTrainingPipeline):
    net: str = "fcos_resnet50_fpn"
    pretrained: bool = True
    collate_method: str = "zip"
    train_batch_size: int = 8
    val_batch_size: int = 1
    lr: float = 0.0001
    optimizer_name: str = "Adam"
    n_gradients: int = 1
    reduce_lr_patience: int = 5

    def create_model(self, dataset_builder) -> Union[torch.nn.Module, LightningModule]:
        num_classes = dataset_builder.num_classes
        model = TorchvisionDetectionModelFactory(
            num_classes=num_classes,
            net=self.net,
            lightning=self.lightning,
            lr=self.lr,  # TODO: lr is specified in 2 places
            pretrained=self.pretrained,
        ).run()
        self.model = model
        return model

    @classmethod
    def model_names(cls):
        return TorchvisionDetectionModelFactory.model_names()
