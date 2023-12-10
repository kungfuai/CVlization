from cvlization.dataset.york_lines import YorkLinesDatasetBuilder
from cvlization.torch.net.line_detection.model_factory import (
    TorchLineDetectionModelFactory,
)
from cvlization.training_pipeline import TrainingPipeline
from cvlization.specs.ml_framework import MLFramework
from cvlization.lab.experiment import Experiment
from cvlization.specs.prediction_tasks import LineDetection
from cvlization.torch.net.line_detection.letr.util import collate_fn


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        self.dataset_builder_cls = YorkLinesDatasetBuilder
        self.model = self.create_model()
        dataset_builder = self.create_dataset()
        training_pipeline = self.create_training_pipeline(self.model)
        Experiment(
            # The interface (inputs and outputs) of the model.
            prediction_task=LineDetection(),
            # Dataset and transforms.
            dataset_builder=dataset_builder,
            # Training pipeline: model, trainer, optimizer.
            training_pipeline=training_pipeline,
        ).run()

    def create_model(self):
        return TorchLineDetectionModelFactory(net=self.args.net).run()

    def create_dataset(self):
        return self.dataset_builder_cls(flavor="torchvision")

    def create_training_pipeline(self, model):
        training_pipeline = TrainingPipeline(
            # Annotating the ml framework helps the training pipeline to use
            #   appropriate adapters for the dataset.
            ml_framework=MLFramework.PYTORCH,
            model=model,
            # Data loader parameters.
            collate_method=collate_fn,
            train_batch_size=2,
            val_batch_size=1,
            # Training loop parameters.
            epochs=500,
            train_steps_per_epoch=100,
            # Optimizer parameters.
            optimizer_name="Adam",
            lr=0.00005,
            n_gradients=2,
            # Experiment tracking/logging.
            experiment_tracker=None,
            # Misc parameters.
            loss_function_included_in_model=True,
        )
        return training_pipeline


if __name__ == "__main__":
    """
    python -m examples.pose_estimation.mmpose.train
    """

    from argparse import ArgumentParser

    options = TorchLineDetectionModelFactory.model_names()
    parser = ArgumentParser(
        epilog=f"""
            *** Options for net: {options} ({len(options)} of them).
            """
    )
    parser.add_argument("--net", type=str, default="letr")
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()
    TrainingSession(args).run()
