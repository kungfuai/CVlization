from dataclasses import dataclass
import logging
from typing import Union
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from ....specs import ModelSpec
from ...torch_model_factory import TorchModelFactory
from ...torch_model import TorchLitModel
from ...encoder.torch_image_encoder import TorchImageEncoder
from ...encoder.torch_image_backbone import create_image_backbone


LOGGER = logging.getLogger(__name__)


@dataclass
class ModelUtils:
    # Model
    model: Union[
        ModelSpec, nn.Module, LightningModule
    ] = None  # If specified, the following parameters will be ignored.
    model_spec: ModelSpec = None
    lightning: bool = True
    loss_function_included_in_model: bool = False  # this is a property of the forward() method of the model

    # Optimizer
    lr: float = 0.0001
    optimizer_name: str = "Adam"
    optimizer_kwargs: dict = None
    lr_scheduler_name: str = None
    lr_scheduler_kwargs: dict = None
    n_gradients: int = 1  # for gradient accumulation
    epochs: int = 10  # Should not be here.

    # Debug
    data_only: bool = False

    def create_model(self, dataset_builder=None):
        if self.data_only:
            # Do nothing.
            return None
        if self._model_spec_is_provided():
            self.model = self.create_model_from_spec()
            LOGGER.info(f"Model created from spec: {type(self.model)}")
            return self.model
        elif self._model_is_provided():
            LOGGER.info(f"Using the model provided by the user: {type(self.model)}")
        else:
            raise ValueError(
                f"model must be a ModelSpec or a Callable object, but got {self.model}"
            )

    def adapt_model(self, model, dataset_builder=None):
        if self.data_only:
            # Do nothing.
            return None
        self.model = self._ensure_torch_lit_model(model)
        return self.model

    def _model_spec_is_provided(self):
        return isinstance(self.model, ModelSpec)

    def _model_is_provided(self):
        return isinstance(self.model, (nn.Module, LightningModule))
    
    def _ensure_torch_lit_model(self, model):
        if model is not None and callable(model):
            if isinstance(model, LightningModule):
                if hasattr(model, "lr"):
                    LOGGER.info(f"Setting lr to {self.lr}")
                    model.lr = self.lr
                else:
                    raise ValueError("Model does not have a lr attribute.")
                return model
            elif isinstance(model, nn.Module):
                LOGGER.info(f"Wrapping model {type(model)} into a LightningModule")
                lit_model = TorchLitModel(
                    config=TorchLitModel.TorchModelConfig(
                        model_inputs=self.get_model_inputs(),
                        model_targets=self.get_model_targets(),
                        model=model,
                        optimizer_name=self.optimizer_name,
                        optimizer_kwargs=self.optimizer_kwargs,
                        n_gradients=self.n_gradients,
                        lr=self.lr,
                        epochs=self.epochs,
                        lr_scheduler_name=self.lr_scheduler_name,
                        lr_scheduler_kwargs=self.lr_scheduler_kwargs,
                    ),
                )
                LOGGER.info(f"{lit_model}")
                return lit_model
            else:
                raise ValueError(
                    f"model must be a LightningModule or torch.nn.Module, got {type(model)}"
                )
        elif model is not None:
            raise ValueError(f"model must be callable, got {type(model)}")
        return model

    def _ensure_loss_function(self):
        if not self.loss_function_included_in_model:
            self._loss_function = self._get_or_create_loss_function(self.model).to(
                self.device
            )
    
    def _get_or_create_loss_function(self, model: TorchLitModel):
        if hasattr(model, "loss_function"):
            return model.loss_function
        else:
            assert self.model_targets is not None, f"model_targets is None but loss_function is not defined in model."
            return TorchModelFactory(
                model_inputs=self.model_spec.get_model_inputs(),
                model_targets=self.model_spec.get_model_targets(),
            ).create_loss_function()

    def _handle_lazy_modules(self, dataloader):
        # TODO: the lazymodule thing should happen during model creation.
        # When using LazyModules, call `forward` with a dummy batch to initialize
        # the parameters before calling torch functions.
        one_batch = next(iter(dataloader))
        inputs, targets = one_batch[0], one_batch[1]
        self.model.eval()
        # TODO: there are multiple ways to call forward(), and it is not clear
        #   which one is intended by the model.
        self.model.forward(inputs)
        # self.model.forward(one_batch)
        self.model.train()  # change back to the training mode
    
    def create_model_from_spec(self):

        model_inputs = self.get_model_inputs()
        model_targets = self.get_model_targets()
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


    def get_model_inputs(self):
        if self.model_spec:
            return self.model_spec.get_model_inputs()

    def get_model_targets(self):
        if self.model_spec:
            return self.model_spec.get_model_targets()
