from dataclasses import dataclass, field
import logging
from typing import List, Callable, Union, Dict

# Try to only import generic python modules.
# from .specs import ModelSpec


LOGGER = logging.getLogger(__name__)

@dataclass
class TrainingPipelineConfig:
    """The config is intented to configure the behavior of the training pipeline,
    as well as being easily tracked in an experiment tracker.
    """
    # ml_framework: MLFramework = MLFramework.PYTORCH
    # Model
    #   Can be a ModelSpec, nn.Module/LightningModule, keras.Model, a python function to transform tensors (for keras)
    # model: Union[
    #     ModelSpec, Callable
    # ] = None  # If specified, the following parameters will be ignored.
    # prediction_task: ModelSpec = None

    loss_function_included_in_model: bool = False

    # Precision
    precision: str = "fp32"  # "fp16", "fp32"

    # Data
    num_workers: int = 0
    collate_method: Union[str, Callable] = None  # "zip", None

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

    # Logging
    experiment_tracker: str = None
    experiment_name: str = "cvlab"
    run_name: str = None

    # Debugging
    data_only: bool = False  # No training, only run through data.


class TrainingPipeline:
    """
    A training pipeline consists of the following components:

    - random seed handling
    - data transforms (and augmentations)
    - dataloaders
    - model creation (architecture, initial weights, etc.) and adaptation (e.g. freeze layers, change strides), sometimes including metrics, losses
    - optimizers
    - trainer and training loop: including validation behavior, callbacks, metrics, losses
    - experiment tracker
    - type and shape checking tools
    - debugging tools

    Each component has a list of knobs to tune. Each knob has a default value and a set of options.
    """

    def __init__(self, config: Union[str, TrainingPipelineConfig, dict] = None):
        self.config = config
        
    def fit(self, dataset_builder):
        LOGGER.info(f"Training pipeline: {self}")
        self._set_random_seed()
        self._prepare_components(dataset_builder)
        if self.experiment_tracker is not None:
            self._log_params()
        try:
            if not self.data_is_compatible(dataset_builder):
                raise ValueError("Dataset is not compatible with the pipeline.")
        except NotImplementedError:
            LOGGER.info("No data compatibility check is implemented for this pipeline. Let's just go ahead.")
        self.model = self._create_model(dataset_builder=dataset_builder)
        self.model = self._adapt_model(self.model, dataset_builder=dataset_builder)

        # dataset transforms need to happen during the following 2 steps.
        train_dataloader = self._create_training_dataloader(dataset_builder)
        val_dataloader = self._create_validation_dataloader(dataset_builder)

        # For tensorflow, treat tf.data.Dataset (assume batch() is already called) as a dataloader.

        trainer = self._create_trainer()
        trainer.fit(self.model, train_dataloader, val_dataloader)
    
    def train(self, dataset_builder):
        """An alias to fit()"""
        return self.fit(dataset_builder)
    
    def data_is_compatible(self, dataset_builder):
        """The runtime check of dataset"""
        raise NotImplementedError
    
    def describe_data_requirement(self):
        """Human readable dataset requirement."""
        ...
    
    # Controller helper methdos:
    def _prepare_components(dataset_builder):
        pass
    
    # Random seed handling.
    def _set_random_seed(self):
        ...

    # Model.
    def _create_model(self, dataset_builder):
        """
        Typically returns:
        - PyTorchLightningModule: already includes train_step, val_step, loss, optimizer.
        - torch.nn.Module: train_step, val_step, loss, optimizer need to be happen in the trainer.
        - Keras.Model: already includes fit, loss, optimizer, if compiled.
        """
        ...
    
    def _adapt_model(self, model, dataset_builder):
        """This is where model surgery can happen.
        Sometimes, this needs to happen continuously, and the trainer may call it multiple times in
        different epochs.

        - Freeze layers.
        - Change strides.
        """
        ...
    
    # Data transforms and augmentations.
    def _transform_training_dataset(self, dataset):
        """
        You can do one or more of the following:
        - Modify the transform(example) function of a Dataset.
        - Modify the __getitem__ function of a Dataset.
        - Modify the collate_fn function of a dataloader.
        - Create a new Dataset class.

        old_transform = dataset.transform
        def new_transform(self, example):
            example = old_transform(example)
            # Do something to the example.

        dataset.transform = new_transform
        return dataset
        """
        ...
    
    def _transform_validation_dataset(self, dataset):
        ...

    # Data loader.
    def _create_training_dataloader(self, dataset_builder):
        """
        Call _transform_training_dataset() here.
        Specify collate_fn here.
        """
        ...
    
    def _create_validation_dataloader(self, dataset_builder):
        """
        Call _transform_validation_dataset() here.
        """
        ...
    
    # Trainer.
    def _create_trainer(self):
        """For torch: Torchlightning trainer.
        For tensorflow: a simple wrapper that just calls keras.Model.fit().
        """
        ...
    
    # Optimizer.
    def _create_optimizers(self, model):
        """If necessary, this can be called in _create_model or _create_trainer.
        """
        ...
    
    ## Experiment tracker.
    def _get_config_dict(self) -> Dict[str, Union[str, int, float]]:
        """Get the config dict to log to the experiment tracker."""
        ...
    
    def _log_params(self):
        ...
    
    def _watch_model(self):
        ...
    
    ## Debugging.
    def _adjust_batch_size_if_doing_data_only_debugging(self):
        ...



class DataReducer:
    """
    R. A. Fisher, 1921: "In its most concrete form, the object of statistical methods is the reduction of data."
    https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.1922.0009

    A DataReducer implements this view, by having one and only method: fit(), which takes
    data and no other arguments as input. No optimizer options, no model architecture options, no training loop options.
    By having this restriction, the DataReducer is encouraged to be self contained recipes, and ready
    to receive data. The user of DataReducers can then focus on getting the data ready.

    This is in contrast to learning systems augmented with causal inference (http://bayes.cs.ucla.edu/BOOK-2K/), where the fit() method
    takes as input not only the data, but also a causal diagram that encodes domain knowledge.
    """
    def fit(self,
        training_dataset=None, validation_dataset=None, test_dataset=None,
        dataset_builder=None,
        data_module=None,
    ):
        raise NotImplementedError
