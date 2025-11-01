from typing import List, Optional
from dataclasses import dataclass
import logging
from torch import nn
import torchmetrics

from ..torch.encoder.torch_mlp_encoder import TorchMlpEncoder
from ..torch.torch_model import TorchLitModel
from .encoder.torch_image_encoder import TorchImageEncoder
from .aggregator.torch_aggregator import TorchAggregator
from ..specs import LossType, DataColumnType, ModelInput, ModelTarget


LOGGER = logging.getLogger(__name__)


@dataclass
class TorchModelCheckpoint:
    model: nn.Module
    epochs_done: int


@dataclass
class TorchModelFactory:
    # TODO: TorchModelFactory may not be needed. Consider using TorchModel to maintain
    #   the class members of TorchModelFactory.
    # TODO: when TorchModelFactory is changed, make it backward compatible.
    #   Otherwise previously trained models may not be loaded successfully.
    """Create customized torch models."""

    # ## Model input and output specs
    model_inputs: List[ModelInput]
    model_targets: List[ModelTarget]

    # ## Model architecture
    model: nn.Module = None
    model_checkpoint_path: str = None

    # ## Optimizer
    optimizer_name: Optional[str] = "Adam"
    optimizer_kwargs: Optional[dict] = None
    lr: Optional[float] = 0.001
    lr_scheduler_name: Optional[str] = None
    lr_scheduler_kwargs: Optional[dict] = None
    n_gradients: Optional[int] = 5
    epochs: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    min_decay: Optional[float] = 0.01

    # ## Checkpointing
    recompile_checkpointed_model: bool = True

    # ## Encoders
    image_encoder: TorchImageEncoder = None
    share_image_encoder: bool = True
    mlp_encoder: TorchMlpEncoder = None

    # ## Aggregators
    aggregator: TorchAggregator = None

    def create_model(self) -> TorchModelCheckpoint:
        return self()

    def __call__(self) -> TorchModelCheckpoint:
        """Create the model."""
        if self.image_encoder is None:
            # self.image_encoder = TorchImageEncoder()
            raise ValueError("Image encoder not set")
        if self.aggregator is None:
            self.aggregator = TorchAggregator()
        model = TorchLitModel(
            TorchLitModel.TorchModelConfig(
                model_inputs=self.model_inputs,
                model_targets=self.model_targets,
                image_encoder=self.image_encoder,
                aggregator=self.aggregator,
                mlp_encoder=self.mlp_encoder,
                share_image_encoder=self.share_image_encoder,
                loss_function=self.create_loss_function(),
                metrics=self.create_metrics(),
                optimizer_name=self.optimizer_name,
                optimizer_kwargs=self.optimizer_kwargs,
                lr=self.lr,
                lr_scheduler_name=self.lr_scheduler_name,
                lr_scheduler_kwargs=self.lr_scheduler_kwargs,
                n_gradients=self.n_gradients,
                epochs=self.epochs,
                steps_per_epoch=self.steps_per_epoch,
                min_decay=self.min_decay,
            )
        )
        return TorchModelCheckpoint(model=model, epochs_done=0)

    def create_metrics(self):
        # It is recommended to use different metric objects for train, val and test.
        # So we create a list of tuples: (metric_class, kwargs)
        # https://torchmetrics.readthedocs.io/en/stable/pages/overview.html
        metrics = []
        for model_target in self.model_targets:
            metrics_for_this_target = []
            # TODO: create metrics based on model_target.metric!
            if model_target.column_type == DataColumnType.BOOLEAN:
                metrics_for_this_target = [
                    (torchmetrics.AUROC, {"num_classes": None, "average": "macro", "task": "BINARY"}),
                    (torchmetrics.AveragePrecision, {"task": "BINARY"}),
                ]
            elif model_target.column_type == DataColumnType.CATEGORICAL:
                metrics_for_this_target = [
                    (
                        torchmetrics.Accuracy,
                        dict(num_classes=model_target.n_categories, average="macro", task="MULTICLASS"),
                    ),
                    (
                        torchmetrics.AUROC,
                        dict(num_classes=model_target.n_categories, average="macro", task="MULTICLASS"),
                    ),
                    # (
                    #     torchmetrics.AveragePrecision,
                    #     dict(
                    #         num_classes=model_target.n_categories,
                    #         average="macro",
                    #     ),
                    # ),
                    # (
                    #     torchmetrics.CohenKappa,
                    #     dict(
                    #         num_classes=model_target.n_categories,
                    #     ),
                    # ),
                ]
            elif model_target.column_type not in [
                DataColumnType.BOOLEAN,
                DataColumnType.CATEGORICAL,
            ]:
                LOGGER.warning(
                    f"Metrics not implemented yet for {model_target.key} of type {model_target.column_type}."
                )
            metrics.append(metrics_for_this_target)
        return metrics

    def create_losses(self):
        loss_functions = []
        loss_weights = []
        for j, model_target in enumerate(self.model_targets):
            loss_weight = (
                1 if model_target.loss_weight is None else model_target.loss_weight
            )
            loss_weights.append(loss_weight)
            if model_target.loss == LossType.CATEGORICAL_CROSSENTROPY:
                loss_fn = nn.CrossEntropyLoss()
            elif model_target.loss == LossType.SPARSE_CATEGORICAL_CROSSENTROPY:
                loss_fn = nn.CrossEntropyLoss()
            elif model_target.loss == LossType.BINARY_CROSSENTROPY:
                loss_fn = nn.BCEWithLogitsLoss()
            elif model_target.loss == LossType.MSE:
                loss_fn = nn.MSELoss()
            elif model_target.loss == LossType.MAE:
                loss_fn = nn.L1Loss()
            else:
                raise NotImplementedError(f"Loss {model_target.loss} not implemented.")
            loss_functions.append(loss_fn)

        for loss_function, loss_weight, model_target in zip(
            loss_functions, loss_weights, self.model_targets
        ):
            LOGGER.info(
                f"adding loss for {model_target.key}: {loss_function}, weight={loss_weight}"
            )

        return loss_functions, loss_weights

    def create_loss_function(self):
        losses, weights = self.create_losses()
        model_targets = self.model_targets

        class CombinedLoss(nn.Module):
            def forward(self, outputs, labels):
                if not isinstance(labels, list):
                    labels = [labels]
                loss_values = []
                for loss_fn, weight, output, label, model_target in zip(
                    losses, weights, outputs, labels, model_targets
                ):
                    if model_target.column_type == DataColumnType.BOOLEAN:
                        # Otherwise BCELoss would complain: RuntimeError: Found dtype Long but expected Float
                        label = label.float()
                        label = label.view(output.shape)
                    LOGGER.debug(
                        f"output: {type(output)}, {output.shape}, {output.dtype}"
                    )
                    LOGGER.debug(
                        f"label: {type(label)}, {label.shape}, {label.dtype}, {label}"
                    )
                    LOGGER.debug(f"weight: {type(weight)}")

                    loss_values.append(loss_fn(output, label) * float(weight))
                total_loss = sum(loss_values)
                return total_loss

        return CombinedLoss()
