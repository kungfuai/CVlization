"""Goal: run computer vision experiments easily.
"""
from typing import Dict

from cvlization.data.dataset_builder import DatasetBuilder


# https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
try:
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    print("Cannot set memory growth for tf")

from .. import Trainer, RichDataFrame, ModelSpec
from .model_specs import ImageClassification
from .datasets import datasets, SplittedDataset, get_dataset_builder_registry
from .training_pipelines import training_pipelines
from ..training_pipeline import TrainingPipeline
from .experiment_tracker import ExperimentTracker


class Experiment:
    """
    Example usage:

    e = Experiment(
        model_spec=Experiment.model_specs()["resnet18"], # model_spec is framework-agnostic (e.g. tf, torch)
        dataset=Experiment.datasets()["cifar10"],
        training_pipeline=Experiment.training_pipelines()["resnet50_tf"],  # trainer is framework-specific
    )
    e.run()
    # e.run_remotely()  # to run in the cloud

    Design goals:
        ModelSpec is an abstract contract about the modeling task. What are the inputs and outputs?
            What are the loss functions? These are declared without specifying how they are implemented.
        Dataset is responsible for instantiating the data.
        Trainer is responsible for instantiating the neural network architecture and training the model.

        The flow of information goes like this:
            ModelSpec -> Dataset -> inputs, targets
            ModelSpec -> Trainer -> model, optimizer, loss, metrics

        The following should be true:
            `inputs` from Dataset should match model inputs.
            `targets` from Dataset should match model outputs, metrics and loss functions.

    # TODO: rename ModelSpec to PredictionTask
    """

    def __init__(
        self,
        prediction_task: ModelSpec,
        dataset_builder: DatasetBuilder,
        training_pipeline: TrainingPipeline,
        experiment_tracker: ExperimentTracker = None,
    ):
        # model_inputs = prediction_task.get_model_inputs()
        # model_targets = prediction_task.get_model_targets()
        self.dataset_builder = dataset_builder
        self.prediction_task = prediction_task
        self.training_pipeline = training_pipeline
        self.experiment_tracker = experiment_tracker
        # TODO: model_inputs and model_targets need to be passed to training_pipeline.

    def run(self):
        # Assemble training pipeline and feed the data.
        if self.experiment_tracker is not None:
            self.experiment_tracker.setup().log_params(self.get_config_dict())
        self.training_pipeline.create_model(self.prediction_task).prepare_datasets(
            self.dataset_builder
        ).create_trainer().run()

    def get_config_dict(self):
        # TODO: flatten the dict if needed.
        return {
            **self.prediction_task.__dict__,
            **self.dataset_builder.__dict__,
            **self.training_pipeline.config.__dict__,
            "framework": self.training_pipeline.framework.value,
        }

    # The catalog of available components are attached to this class, for your convenience.
    @classmethod
    def datasets(cls) -> Dict[str, RichDataFrame]:
        # return datasets()
        return get_dataset_builder_registry()

    @classmethod
    def training_pipelines(cls) -> Dict[str, Trainer]:
        return training_pipelines()


# TODO: remove it
if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES='2' python -m cvlization.lab.experiment
    """
    # ds = Experiment.datasets()["cifar10"]
    # train_ds = ds.training_dataset()
    # for batch in train_ds:
    #     print(batch)
    #     raise ValueError("ok")

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data", "-d", default="cifar10_torchvision")
    parser.add_argument("--model_recipe", "-m", default="resnet18_torch")
    # parser.add_argument("--task", "-t", default="ImageClassification")
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()

    Experiment(
        # The interface (inputs and outputs) of the model.
        prediction_task=ImageClassification(
            n_classes=10, num_channels=3, image_height=32, image_width=32
        ),
        # Dataset and transforms.
        dataset_builder=Experiment.datasets()[args.data.lower()],
        # Model, optimizer and hyperparams.
        training_pipeline=Experiment.training_pipelines()[args.model_recipe],
    ).run()
    # TODO: ExperimentTracker should be an Experiment level config.
