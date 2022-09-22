from dataclasses import dataclass
import logging
from typing import Union, Callable, Optional
import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from ..specs import ModelSpec, MLFramework
from .training_pipeline.utils.dataloader_utils import DataLoaderUtils
from .training_pipeline.utils.model_utils import ModelUtils
from .training_pipeline.utils.trainer_utils import TrainerUtils

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingPipelineConfig:
    pass


@dataclass
class TorchTrainingPipeline:
    """A training pipeline consists of the following components:

    - data transforms (and augmentations)
    - dataloaders
    - torch model
    - optimizers
    - trainer and training loop
    - experiment tracker
    - type and shape checking tools
    - debugging tools

    Each component has a list of knobs to tune. Each knob has a default value and a set of options.

    # TODO: for debugging tools, include near-duplicate detection between train and val.
    """

    ml_framework: MLFramework = MLFramework.PYTORCH

    # Model
    #   Can be a ModelSpec, nn.Module/LightningModule
    model: Union[
        ModelSpec, nn.Module, LightningModule
    ] = None  # If specified, the following parameters will be ignored.
    prediction_task: ModelSpec = None
    lightning: bool = True
    loss_function_included_in_model: bool = False  # this is a property of the forward() method of the model
    
    # Data
    num_workers: int = 0
    collate_method: Union[str, Callable] = "zip"  # "zip", None
    train_batch_size: int = 32
    val_batch_size: int = 32

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

    ## Debugging
    data_only: bool = False  # No training, only run through data.

    def __post_init__(self):
        # add more input args: model config, optimizer config, training_loop_config, validation_config, etc.
        self._adjust_batch_size_if_doing_data_only_debugging()
        # TODO: _populate_model_spec_based_on_user_provided_model is too specific of an
        #   implementation detail. Consider removing it.
        self._populate_model_spec_based_on_user_provided_model()
        assert self.lightning, "PyTorchLighnting is used by default. Please set lightning=True."

    # TODO: define DatasetBuilder inferface
    def fit(self, dataset_builder):
        LOGGER.info(f"Training pipeline: {self}")
        self._prepare_components()
        if self.experiment_tracker is not None:
            self._log_params()
        try:
            if not self.data_is_compatible(dataset_builder):
                raise ValueError("Dataset is not compatible with the pipeline.")
        except NotImplementedError:
            LOGGER.info("No data compatibility check is implemented for this pipeline. Let's just go ahead.")
        self.model = self._create_model(dataset_builder=dataset_builder)
        # This is where model surgery happens.
        self.model = self._adapt_model(self.model, dataset_builder=dataset_builder)
        train_dataloader = self._create_training_dataloader(dataset_builder)
        val_dataloader = self._create_validation_dataloader(dataset_builder)
        trainer = self._create_trainer()
        assert isinstance(self.model, LightningModule), f"Model must be a LightningModule. Got {type(self.model)}."
        trainer.fit(self.model, train_dataloader, val_dataloader)

    def train(self, dataset_builder):
        """An alias for fit().
        """
        return self.fit(dataset_builder)

    def data_is_compatible(self, dataset_builder):
        raise NotImplementedError("This TrainingPipeline does not have specific data requirements.")
    
    def describe_data_requirement(self) -> str:
        # Show examples of data.
        # Display schema (names, types and shapes) of required data.
        raise NotImplementedError("This TrainingPipeline does not have specific data requirements.")
    
    def _create_model(self, dataset_builder):
        return self._model_utils.create_model(dataset_builder=dataset_builder)
    
    def _adapt_model(self, model, dataset_builder):
        return self._model_utils.adapt_model(model, dataset_builder)
    
    def _create_training_dataloader(self, dataset_builder):
        return self._dataloader_utils.create_training_dataloader(dataset_builder)
    
    def _create_validation_dataloader(self, dataset_builder):
        return self._dataloader_utils.create_validation_dataloader(dataset_builder)
    
    def _create_trainer(self):
        return self._trainer_utils.create_trainer()

    # Helper methods:
    def _prepare_components(self):
        self._dataloader_utils = DataLoaderUtils(
            collate_method=self.collate_method,
            num_workers=self.num_workers,
            train_batch_size=self.train_batch_size,
            val_batch_size=self.val_batch_size,
            model_spec=self.model_spec,
        )
        self._model_utils = ModelUtils(
            model=self.model,
            model_spec=self.model_spec,
            lightning=self.lightning,
            loss_function_included_in_model=self.loss_function_included_in_model,
            lr=self.lr,
            optimizer_name=self.optimizer_name,
            optimizer_kwargs=self.optimizer_kwargs,
            lr_scheduler_name=self.lr_scheduler_name,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
            data_only=self.data_only,
        )
        self._trainer_utils = TrainerUtils(
            lr=self.lr,
            n_gradients=self.n_gradients,
            epochs=self.epochs,
            train_batch_size=self.train_batch_size,
            train_steps_per_epoch=self.train_steps_per_epoch,
            val_batch_size=self.val_batch_size,
            val_steps_per_epoch=self.val_steps_per_epoch,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            reduce_lr_patience=self.reduce_lr_patience,
            early_stop_patience=self.early_stop_patience,
            device=self.device,
            auto_select_gpus=self.auto_select_gpus,
            gpus=self.gpus,
            num_nodes=self.num_nodes,
            accelerator=self.accelerator,
            precision=self.precision,
            experiment_tracker=self.experiment_tracker,
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            checkpoint_root_dir=self.checkpoint_root_dir,
            log_every_n_steps=self.log_every_n_steps,
            enable_progress_bar=self.enable_progress_bar,
            data_only=self.data_only,
        )
    
    ## Type and shape checking.

    def _populate_model_spec_based_on_user_provided_model(self):
        if isinstance(self.model, ModelSpec):
            self.model_spec = self.model
        elif isinstance(self.prediction_task, ModelSpec):
            self.model_spec = self.prediction_task
        else:
            self.model_spec = None

    def _get_model_inputs(self):
        if self.model_spec:
            return self.model_spec.get_model_inputs()

    def _get_model_targets(self):
        if self.model_spec:
            return self.model_spec.get_model_targets()

    ## Experiment tracker.
    def _get_config_dict(self):
        raise NotImplementedError("This TrainingPipeline does not have specific config requirements.")

    def _log_params(self):
        self.experiment_tracker.setup().log_params(self._get_config_dict())
    
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

    
    ## Debugging.
    def _adjust_batch_size_if_doing_data_only_debugging(self):
        if self.data_only:
            self.train_batch_size = 1

