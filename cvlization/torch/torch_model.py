from dataclasses import dataclass
import logging
from typing import List, Optional, Any
import torch
from torch import nn, optim
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import Metric

from ..specs import DataColumnType, ModelInput, ModelTarget
from .encoder.torch_image_encoder import TorchImageEncoder
from .encoder.torch_mlp_encoder import TorchMlpEncoder
from .aggregator.torch_aggregator import TorchAggregator


LOGGER = logging.getLogger(__name__)

# https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/5
# eq=False prevents overiding the hash function of nn.Module.


class TorchModel(LightningModule):
    """Default multitask torch model."""

    @dataclass
    class TorchModelConfig:
        # ## Model spec.
        model_inputs: List[ModelInput]
        model_targets: List[ModelTarget]
        image_encoder: TorchImageEncoder = None
        share_image_encoder: bool = True
        mlp_encoder: TorchMlpEncoder = None
        aggregator: TorchAggregator = None
        model: nn.Module = None  # If specified, the above model specs will be ignored.

        # ## Losses and metrics.
        loss_function: nn.Module = None
        metrics: List[List] = None

        # # ## Device
        # device: Optional[torch.device] = (
        #     torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # )

        # ## Training loop
        epochs: Optional[int] = 10
        # initial_epoch set to 1, so that the pre-training eval step can log metrics correctly.
        initial_epoch: Optional[int] = 1
        train_steps_per_epoch: Optional[int] = None
        val_steps_per_epoch: Optional[int] = None

        # ## Optimizer
        optimizer_name: str = "Adam"
        optimizer_kwargs: Optional[dict] = None
        lr: Optional[float] = 0.001
        lr_scheduler_name: Optional[str] = None
        lr_scheduler_kwargs: Optional[dict] = None
        steps_per_epoch: Optional[int] = None
        min_decay: Optional[float] = None
        extra_optimizer_args: Optional[dict] = None
        # TODO: lr_finder not implemented.
        use_lr_finder: Optional[bool] = False
        n_gradients: int = (
            1  # Number of steps for gradient accumulation before updating the weights.
        )

        # ## Info
        name: Optional[str] = "torch_trainer"

        # ## Tracking
        experiment_tracker: Optional[Any] = None

    def __init__(self, config: TorchModelConfig):
        super().__init__()
        self.config = config

        self._metrics = {}
        for dataset_prefix in ["train_", "val_", "test_"]:
            self._metrics[dataset_prefix] = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            metric_class(**kwargs)
                            for metric_class, kwargs in metrics_for_one_target
                        ]
                    )
                    for metrics_for_one_target in self.config.metrics
                ]
            )
        self._metrics = nn.ModuleDict(self._metrics)
        # TODO: should the following creation of actual layers wait for the first forward pass?
        if isinstance(self.config.model, nn.Module):
            LOGGER.warning(
                "Model is already provided by the user. Ignoring model components and model spec."
            )
            self.model = self.config.model
        else:
            self._create_mlp_encoder_if_needed()
            self._create_aggregator_if_needed()
            self._prepare_encoder_models()
            self._create_heads()

    def forward(self, inputs):
        if isinstance(self.config.model, nn.Module):
            return self.model.forward(inputs)
        tensors_encoded = []
        tensors_not_encoded = []
        for input_layer, encoder_model in zip(inputs, self._encoder_models):
            if encoder_model is not None:
                encoded = encoder_model(input_layer.float())
                tensors_encoded.append(encoded)
            else:
                tensors_not_encoded.append(input_layer)

        if len(tensors_not_encoded) > 1:
            mlp_input_tensor = torch.cat(tensors_not_encoded, dim=-1)
        elif len(tensors_not_encoded) == 1:
            mlp_input_tensor = tensors_not_encoded[0]
        else:
            mlp_input_tensor = None

        if mlp_input_tensor is not None:
            encoded_by_mlp = self._mlp_encoder(mlp_input_tensor)
            tensors_encoded.append(encoded_by_mlp)

        encoded_agg = self._aggregator(tensors_encoded)

        output_tensors = []
        for model_target, head_module in zip(self.config.model_targets, self._heads):
            output_tensor = head_module(encoded_agg)
            if model_target.column_type == DataColumnType.BOOLEAN:
                # BCE expects the output shape to be [batch_size].
                output_tensor = torch.squeeze(output_tensor, dim=-1)
            output_tensors.append(output_tensor)

        return output_tensors

    def _create_mlp_encoder_if_needed(self):
        if self.config.mlp_encoder is None:
            self._mlp_encoder = TorchMlpEncoder()
        else:
            self._mlp_encoder = self.config.mlp_encoder
        self._mlp_encoder

    def _create_aggregator_if_needed(self):
        if self.config.aggregator is None:
            self._aggregator = TorchAggregator()
        else:
            self._aggregator = self.config.aggregator

    def _prepare_encoder_models(self):
        _encoder_models = []
        if self.config.share_image_encoder:
            self._shared_image_encoder = self.config.image_encoder
        for model_input in self.config.model_inputs:
            if model_input.column_type == DataColumnType.IMAGE:
                if self.config.share_image_encoder:
                    # Appending the reference to the encoder.
                    _encoder_models.append(self._shared_image_encoder)
                else:
                    raise NotImplementedError("Need to pass in different encoders.")
            elif model_input.column_type in [
                DataColumnType.NUMERICAL,
                DataColumnType.CATEGORICAL,
                DataColumnType.BOOLEAN,
            ]:
                _encoder_models.append(None)
            else:
                raise NotImplementedError(
                    f"Data column type {model_input.column_type} not supported."
                )
        self._encoder_models = nn.ModuleList(_encoder_models)

    def _load_model_checkpoint(self, checkpoint_path: str):
        pass

    def _create_heads(self):
        self._heads = []
        for model_target in self.config.model_targets:
            if model_target.column_type == DataColumnType.BOOLEAN:
                self._heads.append(self._create_binary_classifier())
            elif model_target.column_type == DataColumnType.NUMERICAL:
                self._heads(self._create_regressor())
            elif model_target.column_type == DataColumnType.CATEGORICAL:
                classifier = self._create_multiclass_classifier(
                    n_classes=model_target.n_categories
                )
                self._heads.append(classifier)
        self._heads = nn.ModuleList(self._heads)

    def _create_binary_classifier(self):
        return nn.Sequential(
            nn.LazyLinear(1),
            nn.Sigmoid(),
        )

    def _create_multiclass_classifier(self, n_classes: int):
        return nn.Sequential(
            nn.LazyLinear(n_classes),
            # nn.Softmax(dim=-1),
        )

    def _create_regressor(self, out_dim: int = 1):
        return nn.LazyLinear(out_dim)

    def reset_metrics(self, dataset_prefix: str):
        for metrics_for_one_target in self._metrics[dataset_prefix]:
            for m in metrics_for_one_target:
                m.reset()

    def training_step(self, batch, batch_idx):
        if len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            raise ValueError(f"Batch has {len(batch)} parts.")
        outputs = self(inputs)
        loss = self.config.loss_function(outputs, targets)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True
        )  # optional args: on_step=False, on_epoch=True
        self._update_metrics(outputs, targets, dataset_prefix="train_")
        return {"loss": loss}

    def _get_optimizer_class(self):
        optimizer_name = self.config.optimizer_name
        if optimizer_name == "SGD_david":
            # from .net.davidnet.torch_backend import SGD
            # return SGD
            return optim.SGD
        if isinstance(optimizer_name, str) and hasattr(optim, optimizer_name):
            return getattr(optim, optimizer_name)
        LOGGER.warning(f"Cannot find optimizer {optimizer_name}. Using Adam.")
        return optim.Adam

    def configure_optimizers(self) -> dict:
        create_optimizer = self._get_optimizer_class()
        optimizer_kwargs = self.config.optimizer_kwargs or {}
        if self.config.optimizer_name == "SGD_david":
            from .net.davidnet.core import PiecewiseLinear, Const

            epochs = 20
            lr_schedule = PiecewiseLinear([0, 5, epochs], [0, 0.4, 0])
            batch_size = 512
            n_train_batches = int(50000 / batch_size)
            optimizer_kwargs = {
                "lr": 0.01,
                "weight_decay": Const(5e-4 * batch_size),
                "momentum": Const(0.9),
            }
            lr_schedule_func = (
                lambda step: lr_schedule(step / n_train_batches) / batch_size
            )
            opt = create_optimizer(self.parameters(), **optimizer_kwargs)
            return {"optimizer": opt}
        else:
            opt = create_optimizer(self.parameters(), **optimizer_kwargs)
        optimizer_dict = {"optimizer": opt}
        lr_scheduler_name = self.config.lr_scheduler_name
        if lr_scheduler_name is not None:
            lr_scheduler_class = getattr(optim.lr_scheduler, lr_scheduler_name)
            assert (
                lr_scheduler_class is not None
            ), f"Cannot find lr_scheduler {lr_scheduler_name}"
            lr_scheduler = lr_scheduler_class(opt, **self.config.lr_scheduler_kwargs)
            scheduler_dict = {"scheduler": lr_scheduler, "interval": "step"}
            optimizer_dict["lr_scheduler"] = scheduler_dict

        return optimizer_dict

    def test_step(self, batch, batch_idx):
        if len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            raise ValueError(f"Batch has {len(batch)} parts.")
        outputs = self(inputs)
        loss = self.config.loss_function(outputs, targets)
        self.log("test_loss", loss)
        self._update_metrics(outputs, targets, dataset_prefix="test_")
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            raise ValueError(f"Batch has {len(batch)} parts.")
        outputs = self(inputs)
        loss = self.config.loss_function(outputs, targets)
        self.log("val_loss", loss)
        self._update_metrics(outputs, targets, dataset_prefix="val_")
        return loss

    def _update_metrics(self, outputs: list, targets: list, dataset_prefix: str):
        for i_target, metrics_for_one_target in enumerate(
            self._metrics[dataset_prefix]
        ):
            for m in metrics_for_one_target:
                m.update(outputs[i_target], targets[i_target])

    def _compute_and_log_metrics(self, dataset_prefix: str, reset=True):
        all_metric_values = {}
        for metrics_for_one_target in self._metrics[dataset_prefix]:
            for m in metrics_for_one_target:
                metric_value = m.compute()
                metric_name = dataset_prefix + type(m).__name__
                self.log(metric_name, metric_value, prog_bar=False, on_epoch=True)
                if hasattr(metric_value, "cpu"):
                    metric_value = metric_value.cpu()
                metric_value = metric_value.detach().numpy()
                try:
                    metric_value = float(metric_value)
                except:
                    pass
                all_metric_values[metric_name] = metric_value
        print("\n==============================================")
        print(all_metric_values)
        if reset:
            self.reset_metrics(dataset_prefix)

    def on_train_epoch_end(self):
        self._compute_and_log_metrics(dataset_prefix="train_")
        super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        self._compute_and_log_metrics(dataset_prefix="val_")
        super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        self._compute_and_log_metrics(dataset_prefix="test_")
        super().on_test_epoch_end()
