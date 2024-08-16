from dataclasses import dataclass
import logging
from typing import Union, Callable, Optional
import torch
from torch import nn
from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import callbacks as cb
from ....specs import ModelSpec

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainerUtils:
    # Optimizer
    lr: float = 0.0001
    optimizer_name: str = "Adam"
    optimizer_kwargs: dict = None
    lr_scheduler_name: str = None
    lr_scheduler_kwargs: dict = None
    n_gradients: int = 1  # for gradient accumulation
    epochs: int = 10
    train_batch_size: int = 32
    train_steps_per_epoch: int = None
    val_batch_size: int = 32
    val_steps_per_epoch: int = None
    check_val_every_n_epoch: int = 5
    reduce_lr_patience: int = 5
    early_stop_patience: int = 10

    ## Device
    device: Optional[torch.device] = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    auto_select_gpus: bool = True
    gpus: int = 1 if torch.cuda.is_available() else 0
    num_nodes: int = 1
    accelerator: str = None
    precision: int = 32

    ## Experiment tracking and logging
    experiment_tracker: str = None
    experiment_name: str = "torch_training"
    run_name: str = None
    checkpoint_root_dir: Optional[str] = None
    log_every_n_steps: int = 10
    enable_progress_bar: bool = True

    ## Debug
    data_only: bool = False

    def create_trainer(self):
        if self.data_only:
            return None

        limit_train_batches = self.train_steps_per_epoch or 1.0
        limit_val_batches = self.val_steps_per_epoch or 1.0
        LOGGER.info(f"limit train batches: {limit_train_batches}")
        LOGGER.info(f"limit val batches: {limit_val_batches}")

        if not self.checkpoint_root_dir:
            weights_save_path = None
            enable_checkpointing = False
            LOGGER.info("Disable checkpointing in pytorch-lightning.")
        else:
            weights_save_path = self.checkpoint_root_dir
            enable_checkpointing = True
        
        logger_for_experiment_tracking = self.create_experiment_tracker()

        trainer = Trainer(
            # TODO: RuntimeError: No GPUs available.
            gpus=self.gpus,
            auto_select_gpus=self.auto_select_gpus,
            max_epochs=self.epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            default_root_dir=self.checkpoint_root_dir,
            # weights_save_path=weights_save_path,
            # checkpoint_callback=enable_checkpointing,
            enable_checkpointing=enable_checkpointing,
            logger=logger_for_experiment_tracking,
            # TODO: https://github.com/PyTorchLightning/pytorch-lightning/issues/6170
            #    lr scheduler may interfere with grad accumulation.
            accumulate_grad_batches=self.n_gradients,
            precision=self.precision,
            num_nodes=self.num_nodes,
            accelerator=self.accelerator,
            callbacks=self.create_callbacks(),
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.log_every_n_steps,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
        )
        self.trainer = trainer
        return trainer
    
    def create_callbacks(self):
        callbacks = [
            # cb.EarlyStopping(
            #     patience=self.early_stopping_patience,
            #     monitor="val_loss",
            #     mode="min",
            # ),
        ]
        if self.experiment_tracker is not None:
            callbacks.append(
                cb.LearningRateMonitor(
                    logging_interval="epoch",
                    log_momentum=True,
                )
            )
        return callbacks
    
    def create_experiment_tracker(self):
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
