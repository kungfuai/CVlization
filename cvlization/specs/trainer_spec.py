from dataclasses import dataclass
from typing import List, Callable, Union

from .data_column import DataColumnType
from . import ModelSpec, MLFramework


@dataclass
class TrainerSpec:
    ml_framework: MLFramework = MLFramework.PYTORCH
    provider: str = None

    # Model
    #   Can be a ModelSpec, nn.Module/LightningModule, keras.Model, a python function to transform tensors (for keras)
    model: Union[
        ModelSpec, Callable
    ] = None  # If specified, the following parameters will be ignored.
    loss_function_included_in_model: bool = False

    # Precision
    precision: str = "fp32"  # "fp16", "fp32"

    # Data
    num_workers: int = 0
    collate_method: str = None  # "zip", None

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
    reduce_lr_patience: int = 5
    early_stop_patience: int = 10

    # Logging
    experiment_tracker: str = None
    experiment_name: str = "cvlab"
    run_name: str = None

    # Debugging
    data_only: bool = False  # No training, only run through data.
