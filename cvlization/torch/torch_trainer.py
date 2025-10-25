from dataclasses import dataclass
import numpy as np
from typing import Union, Optional, Any, Iterable, List, Callable
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import logging
import os
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
)

from ..data.dataset_builder import MapStyleDataset, Iterable, Dataset
from ..data.ml_dataset import MLDataset, ModelInput, ModelTarget
from .torch_model import TorchLitModel
from .torch_dataset import MapDataset, GeneratorDataset
from .torch_model_factory import TorchModelFactory
from ..base_trainer import BaseTrainer


LOGGER = logging.getLogger(__name__)


@dataclass
class TorchTrainer(BaseTrainer):

    # ## Model (a trainable)
    model: Union[TorchLitModel, nn.Module]
    model_inputs: List[ModelInput]
    model_targets: List[ModelTarget]

    # ## Datasets
    train_dataset: Dataset
    val_dataset: Dataset = None
    train_batch_size: int = 1
    val_batch_size: int = 1
    collate_method: Union[
        str, Callable
    ] = None  # "zip", None (None means default collate_fn)
    num_workers: int = 0

    loss_function_included_in_model: bool = False

    # ## Precision
    precision: Optional[str] = "fp32"

    # ## Checkpoint directory
    log_dir: str = os.getcwd()  # TODO: this is not used.

    # ## Device
    device: Optional[torch.device] = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # ## Training loop
    epochs: Optional[int] = 10
    # initial_epoch set to 1, so that the pre-training eval step can log metrics correctly.
    initial_epoch: Optional[int] = 1
    train_steps_per_epoch: Optional[int] = None
    val_steps_per_epoch: Optional[int] = None
    check_val_every_n_epoch: Optional[int] = 5  # Used for Pytorch Lightning Trainer.

    # TODO: lr_finder not implemented.
    use_lr_finder: Optional[bool] = False
    n_gradients: int = (
        1  # Number of steps for gradient accumulation before updating the weights.
    )
    early_stopping_patience: int = 20

    # ## Info
    name: Optional[str] = "torch_trainer"

    # ## Tracking
    experiment_tracker: Optional[Any] = None
    experiment_name: Optional[
        str
    ] = "cvlab"  # this is the project name in wandb, experiment name in mlflow
    run_name: Optional[str] = None
    checkpoint_root_dir: Optional[str] = None

    def __post_init__(self):
        assert isinstance(self.model, LightningModule)
        self._validate_fields()
        super().__init__(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            experiment_tracker=self.experiment_tracker,
        )
        limit_train_batches = self.train_steps_per_epoch or 1.0
        limit_val_batches = self.val_steps_per_epoch or 1.0

        LOGGER.info(f"limit train batches: {limit_train_batches}")
        LOGGER.info(f"limit val batches: {limit_val_batches}")

        if not self.log_dir:
            enable_checkpointing = False
            LOGGER.info("Disable checkpointing in pytorch-lightning.")
        else:
            enable_checkpointing = True

        logger_for_experiment_tracking = self._set_up_experiment_tracker()
        callbacks = [
            # EarlyStopping(
            #     patience=self.early_stopping_patience,
            #     monitor="val_loss",
            #     mode="min",
            # ),
        ]
        if self.experiment_tracker is not None:
            callbacks.append(
                LearningRateMonitor(
                    logging_interval="epoch",
                    log_momentum=True,
                )
            )
        self._lightning_trainer = Trainer(
            # Lightning 2.x: use accelerator and devices instead of gpus
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
            max_epochs=self.epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            default_root_dir=self.checkpoint_root_dir,
            enable_checkpointing=enable_checkpointing,
            logger=logger_for_experiment_tracking,
            # TODO: https://github.com/PyTorchLightning/pytorch-lightning/issues/6170
            #    lr scheduler may interfere with grad accumulation.
            accumulate_grad_batches=self.n_gradients,
            precision=16 if self.precision == "fp16" else 32,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=10,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
        )
        if not self.loss_function_included_in_model:
            self._loss_function = self._get_or_create_loss_function(self.model).to(
                self.device
            )

    def log_params(self, params: dict):
        if self.experiment_tracker == "wandb":
            self._logger_for_experiment_tracking.experiment.config.update(params)
        elif self.experiment_tracker is not None:
            LOGGER.info(
                f"Logging params for {self.experiment_tracker} not implemented yet."
            )

    def _set_up_experiment_tracker(self):
        if self.experiment_tracker == "wandb":
            from pytorch_lightning.loggers import WandbLogger

            # TODO: for wandb, customize the summary method.
            # define a metric we are interested in the maximum of
            # e.g. wandb.define_metric("acc", summary="max")
            #
            # MLFlow, on the other hand, does not support this yet:
            #   https://github.com/mlflow/mlflow/issues/4750

            self._logger_for_experiment_tracking = WandbLogger(
                project=self.experiment_name, experiment=self.run_name, save_dir="wandb"
            )
        elif self.experiment_tracker == "mlflow":
            raise NotImplementedError("mlflow logger not implemented yet.")
        elif self.experiment_tracker is None:
            self._logger_for_experiment_tracking = None
            LOGGER.info("No experiment tracking.")
        else:
            raise ValueError(f"Unknown experiment tracker: {self.experiment_tracker}")
        return self._logger_for_experiment_tracking

    def _watch_model(self):
        if self.experiment_tracker == "wandb":
            if isinstance(self.model, nn.Module) or isinstance(
                self.model, LightningModule
            ):
                self._logger_for_experiment_tracking.watch(self.model)
            else:
                raise ValueError(f"Cannot watch model: {self.model}")
        elif self.experiment_tracker is not None:
            LOGGER.info(
                f"Watching model for {self.experiment_tracker} not implemented yet."
            )

    def create_collate_fn(self):
        if self.collate_method == "zip":

            def collate_fn(batch):
                return tuple(zip(*batch))

            return collate_fn
        elif callable(self.collate_method):
            return self.collate_method
        else:
            return None

    def _training_loop(self):
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset
        train_batch_size = self.train_batch_size
        val_batch_size = self.val_batch_size
        num_workers = self.num_workers
        collate_fn = self.create_collate_fn()

        class DataModule(LightningDataModule):
            def train_dataloader(self):
                if isinstance(train_dataset, DataLoader):
                    return train_dataset
                # Error: "Subscripted generics cannot be used with class and instance checks."
                elif isinstance(train_dataset, MapStyleDataset):
                    return DataLoader(
                        train_dataset,
                        batch_size=train_batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        collate_fn=collate_fn,
                    )
                elif isinstance(train_dataset, Iterable):
                    return DataLoader(
                        train_dataset,
                        batch_size=train_batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        collate_fn=collate_fn,
                    )
                else:
                    raise ValueError(f"Unknown train dataset: {train_dataset}")

            def val_dataloader(self):
                if isinstance(val_dataset, DataLoader):
                    return val_dataset
                elif isinstance(val_dataset, MapStyleDataset) or isinstance(
                    val_dataset, Iterable
                ):
                    return DataLoader(
                        val_dataset,
                        batch_size=val_batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        collate_fn=collate_fn,
                    )
                else:
                    raise ValueError(f"Unknown val dataset: {val_dataset}")

            def test_dataloader(self):
                return None

        dm = DataModule()
        if isinstance(train_dataset, MLDataset):
            LOGGER.info("loading some data ...")
            d = DataLoader(
                MapDataset(val_dataset),
                batch_size=val_dataset.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            d = [MapDataset(val_dataset)[1]]
            for b in d:
                print(len(b), type(b))
                break

        # TODO: the lazymodule thing should happen during model creation.
        if True:
            # When using LazyModules, call `forward` with a dummy batch to initialize
            # the parameters before calling torch functions.
            one_batch = next(iter(dm.train_dataloader()))
            inputs, targets = one_batch[0], one_batch[1]
            self.model.eval()
            # TODO: there are multiple ways to call forward(), and it is not clear
            #   which one is intended by the model.
            self.model.forward(inputs)
            # self.model.forward(one_batch)
            self.model.train()  # change back to the training mode

        self._watch_model()

        LOGGER.info(f"optimizer: {self.model.configure_optimizers()}")
        self._lightning_trainer.fit(self.model, dm)

    def get_metrics(self) -> dict:
        return self._lightning_trainer.callback_metrics

    def _get_optimizer_class(self):
        if isinstance(self.optimizer_name, str) and hasattr(optim, self.optimizer_name):
            return getattr(optim, self.optimizer_name)
        LOGGER.warning(f"Cannot find optimizer {self.optimizer_name}. Using Adam.")
        return optim.Adam

    def _get_current_epoch(self, checkpoint_dir: str) -> int:
        raise NotImplementedError("Read epoch info from checkpoint_dir")

    def _get_or_create_loss_function(self, model: TorchLitModel):
        if hasattr(model, "loss_function"):
            return model.loss_function
        else:
            assert self.model_targets is not None, f"model_targets is None but loss_function is not defined in model."
            return TorchModelFactory(
                model_inputs=self.model_inputs,
                model_targets=self.model_targets,
            ).create_loss_function()

    def _validate_fields(self):
        if self.model is None:
            raise ValueError("Please set the model field to a TorchModel.")
        if not self.n_gradients >= 1:
            raise ValueError("n_gradients must be >= 1")
        self.n_gradients = int(self.n_gradients)
