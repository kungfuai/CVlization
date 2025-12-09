from cvlization.dataset.york_lines import YorkLinesDatasetBuilder
from cvlization.torch.net.line_detection.model_factory import (
    TorchLineDetectionModelFactory,
)
from cvlization.cross_framework_training_pipeline import CrossFrameworkTrainingPipeline
from cvlization.specs.ml_framework import MLFramework
from cvlization.torch.net.line_detection.letr.util import collate_fn


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        self.dataset_builder_cls = YorkLinesDatasetBuilder
        self.model = self.create_model()
        dataset_builder = self.create_dataset()
        training_pipeline = self.create_training_pipeline(self.model)
        training_pipeline.create_model().create_dataloaders(
            dataset_builder
        ).create_trainer().run()

    def create_model(self):
        return TorchLineDetectionModelFactory(net=self.args.net).run()

    def create_dataset(self):
        return self.dataset_builder_cls(
            flavor="torchvision",
            data_dir="/root/.cache/cvlization/data/york_lines"
        )

    def create_training_pipeline(self, model):
        training_pipeline = CrossFrameworkTrainingPipeline(
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
