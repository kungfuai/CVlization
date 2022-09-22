from dataclasses import dataclass, field
import logging
from typing import List, Callable, Union

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

    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        
    def fit(self, dataset_builder):
        self._prepare_components(dataset_builder)
    
    def train(self, dataset_builder):
        """An alias to fit()"""
        return self.fit(dataset_builder)
    
    def data_is_compatible(self, dataset_builder):
        ...
    
    def describe_data_requirement(self):
        ...
    
    # Controller helper methdos:
    def _prepare_components(dataset_builder):
        pass
    
    def _create_model(self, dataset_builder):
        ...
    
    def _adapt_model(self, model, dataset_builder):
        ...
    
    def _create_training_dataloader(self, dataset_builder):
        ...
    
    def _create_validation_dataloader(self, dataset_builder):
        ...
    
    def _create_trainer(self):
        ...
    
    ## Experiment tracker.
    def _get_config_dict(self):
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

    A DataReducer implements this view, by having one and only method: _reduce, which takes
    data and no other arguments as input. No optimizer options, no model architecture options, no training loop options.
    By having this restriction, the DataReducer is encouraged to be self contained recipes, and ready
    to receive data. The user of DataReducers can then focus on getting the data ready.

    This is in contrast to learning systems augmented with causal inference (http://bayes.cs.ucla.edu/BOOK-2K/), where the reduce() method
    takes as input not only the data, but also a causal diagram that encodes domain knowledge.
    """
    def _reduce(self,
        training_dataset=None, validation_dataset=None, test_dataset=None,
        dataset_builder=None,
        data_module=None,
    ):
        raise NotImplementedError
