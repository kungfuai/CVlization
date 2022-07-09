# Adapted from https://github.com/bmild/nerf
import logging
import tensorflow as tf
from tensorflow import keras

from cvlization.tensorflow.net.nerf.tiny_nerf import TinyNerfModel
from cvlization.specs.ml_framework import MLFramework
from cvlization.specs.prediction_tasks.nerf import Nerf
from cvlization.training_pipeline import TrainingPipeline
from cvlization.lab.experiment import Experiment
from cvlization.lab.tiny_nerf import TinyNerfDatasetBuilder
from cvlization.tensorflow.metrics.psnr import PSNR


LOGGER = logging.getLogger(__name__)


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        prediction_task = Nerf(
            n_channels=3,
            channels_first=False,
        )

        model = self.create_model()

        training_pipeline = TrainingPipeline(
            ml_framework=MLFramework.TENSORFLOW,
            prediction_task=prediction_task,
            model=model,
            loss_function_included_in_model=False,
            collate_method=None,
            epochs=50,
            train_batch_size=1,
            val_batch_size=1,
            train_steps_per_epoch=10,
            val_steps_per_epoch=None,
            optimizer_name="Adam",
            lr=0.0001,
            n_gradients=1,
            experiment_tracker=None,
        )

        Experiment(
            # The interface (inputs and outputs) of the model.
            prediction_task=prediction_task,
            # Dataset and transforms.
            dataset_builder=TinyNerfDatasetBuilder(),
            # Training pipeline: model, trainer, optimizer.
            training_pipeline=training_pipeline,
        ).run()

    def create_model(self) -> keras.Model:
        model = TinyNerfModel()
        optimizer = keras.optimizers.Adam(5e-4)
        model.compile(
            optimizer=optimizer,
            metrics=[keras.metrics.MeanSquaredError()],
            loss=keras.losses.MeanSquaredError(),
            run_eagerly=True,
        )
        return model


if __name__ == "__main__":
    """
    python -m examples.nerf.tf.train
    """

    from argparse import ArgumentParser

    # options = ["nerf"]
    parser = ArgumentParser(
        # epilog=f"""
        #         options for net: {options} ({len(options)} of them).
        #     """
    )
    # parser.add_argument("--track", action="store_true")
    # parser.add_argument("--net", default="resnet18")
    args = parser.parse_args()
    TrainingSession(args).run()
