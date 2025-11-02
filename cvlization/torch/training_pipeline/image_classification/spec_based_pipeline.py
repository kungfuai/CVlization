import logging
from dataclasses import dataclass
from torch.utils.data import DataLoader
from ...lightning_utils import LightningModule

# TODO: specs are useful for multi-task model, consider not importing them
from cvlization.specs.ml_framework import MLFramework
from cvlization.specs import ModelSpec
from cvlization.specs.prediction_tasks import ImageClassification

# TODO: consider not importing this dataset builder
from cvlization.torch.data.torchvision_dataset_builder import TorchvisionDatasetBuilder

from cvlization.torch.encoder.torch_image_backbone import image_backbone_names, create_image_backbone
from cvlization.torch.torch_training_pipeline import TorchTrainingPipeline
from cvlization.torch.encoder.torch_image_encoder import TorchImageEncoder
from cvlization.torch.torch_model import TorchLitModel
from cvlization.torch.torch_model_factory import TorchModelFactory


LOGGER = logging.getLogger(__name__)

@dataclass
class SpecBasedImageClassification(TorchTrainingPipeline):
    """This is a multi-task training pipeline based on ModelSpec.
    You can use multiple inputs and multiple outputs
    by specifying the ModelSpec.
    """
    net: str = "resnet18"
    num_classes: int = None
    num_channels: int = None
    image_height: int = None
    image_width: int = None
    channels_first: bool = True
    loss_function_included_in_model: bool = False
    collate_method = None
    epochs: int = 10
    train_batch_size: int = 128
    val_batch_size: int = 128
    train_steps_per_epoch: int = None
    val_steps_per_epoch: int = None
    optimizer_name: str = "Adam"
    lr: float = 0.0001
    n_gradients: int = 1  # for gradient accumulation
    experiment_tracker = None

    def train(self, dataset_builder=TorchvisionDatasetBuilder(dataset_classname="CIFAR10")):
        """Alias for fit()."""
        return self.train(dataset_builder=dataset_builder)
    
    def fit(self, dataset_builder=TorchvisionDatasetBuilder(dataset_classname="CIFAR10")):
        if self.model is None:
            prediction_task = ImageClassification(
                n_classes=self.num_classes or dataset_builder.num_classes,
                num_channels=self.num_channels or dataset_builder.num_channels,
                image_height=self.image_height,
                image_width=self.image_width,
                channels_first=self.channels_first,
            )
            self.model = self.model_spec = ModelSpec(
                image_backbone=self.net,
                model_inputs=prediction_task.get_model_inputs(),
                model_targets=prediction_task.get_model_targets(),
            )

        return super().fit(dataset_builder=dataset_builder)
    
    def _create_model(self, dataset_builder):
        if self.model_spec:
            return self._create_torch_model_from_spec(self.model_spec)
        else:
            return super()._create_model(dataset_builder)
    
    def _create_training_dataloader(self, dataset_builder):
        return DataLoader(dataset_builder.training_dataset(), batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)
    
    def _create_validation_dataloader(self, dataset_builder):
        return DataLoader(dataset_builder.validation_dataset(), batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)

    def _create_torch_model_from_spec(self, model_spec):
        model_inputs = model_spec.get_model_inputs()
        model_targets = model_spec.get_model_targets()
        image_encoder = TorchImageEncoder(
            backbone=create_image_backbone(
                self.model_spec.image_backbone,
                pretrained=self.model_spec.pretrained,
                in_chans=self.model_spec.input_shape[-1],
            ),
            permute_image=self.model_spec.permute_image,
            customize_conv1=self.model_spec.customize_conv1,
            dense_layer_sizes=self.model_spec.dense_layer_sizes,
        )
        ckpt = TorchModelFactory(
            model_inputs=model_inputs,
            model_targets=model_targets,
            image_encoder=image_encoder,
            optimizer_name=self.optimizer_name,
            optimizer_kwargs=self.optimizer_kwargs,
            n_gradients=self.n_gradients,
            lr=self.lr,
            epochs=self.epochs,
            lr_scheduler_name=self.lr_scheduler_name,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
        ).create_model()
        model = ckpt.model

        if not isinstance(model, LightningModule):
            model = TorchLitModel(
                config=TorchLitModel.TorchModelConfig(
                    model_inputs=self.get_model_inputs(),
                    model_targets=self.get_model_targets(),
                    model=self.model,
                    optimizer_name=self.optimizer_name,
                    optimizer_kwargs=self.optimizer_kwargs,
                    n_gradients=self.n_gradients,
                    lr=self.lr,
                    epochs=self.epochs,
                    lr_scheduler_name=self.lr_scheduler_name,
                    lr_scheduler_kwargs=self.lr_scheduler_kwargs,
                ),
            )
        assert isinstance(model, LightningModule)
        return model

