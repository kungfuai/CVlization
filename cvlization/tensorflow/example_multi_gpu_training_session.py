import atexit
from dataclasses import dataclass
import logging
import os
from typing import Any

# https://www.tensorflow.org/api_docs/python/tf/autograph/set_verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
import tensorflow as tf

tf.get_logger().setLevel("WARNING")
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50

LOGGER = logging.getLogger(__name__)
# tf.config.run_functions_eagerly(True)


@dataclass
class ExampleMultiGpuTrainingSession:
    data_name: str = "mock_image"
    model_name: str = "resnet50"
    strategy: str = None
    batch_size: int = 4
    epochs: int = 3
    num_mocked_examples: int = 100

    training_dataset: Any = None
    validation_dataset: Any = None
    num_classes: int = 10

    def run(self):
        # from https://keras.io/guides/distributed_training/
        def get_simple_model():
            # Make a simple 2-layer densely-connected neural network.
            inputs = keras.Input(shape=(784,))
            x = keras.layers.Dense(256, activation="relu")(inputs)
            x = keras.layers.Dense(256, activation="relu")(x)
            outputs = keras.layers.Dense(self.num_classes)(x)
            model = keras.Model(inputs, outputs)
            model.compile(
                optimizer=keras.optimizers.Adam(),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.SparseCategoricalAccuracy()],
            )
            return model

        def get_small_conv_model():
            inputs = keras.Input(shape=(None, None, 3))
            x = keras.layers.Conv2D(32, 3, activation="relu")(inputs)
            x = keras.layers.Conv2D(32, 3, activation="relu")(x)
            x = keras.layers.GlobalMaxPool2D()(x)
            outputs = keras.layers.Dense(self.num_classes)(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=keras.optimizers.Adam(),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.SparseCategoricalAccuracy()],
            )
            return model

        def get_resnet50():
            backbone = ResNet50(include_top=False, weights=None, pooling="avg")
            model = keras.models.Sequential(
                [backbone, keras.layers.Dense(self.num_classes)]
            )
            model.compile(
                optimizer=keras.optimizers.Adam(),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.SparseCategoricalAccuracy()],
            )
            return model

        def get_model():
            if self.model_name == "simple":
                return get_simple_model()
            elif self.model_name == "small_conv":
                return get_small_conv_model()
            elif self.model_name == "resnet50":
                return get_resnet50()
            else:
                raise NotImplementedError

        def get_dataset():
            if self.training_dataset:
                return self.training_dataset, self.validation_dataset, None
            if self.data_name == "mnist":
                return get_mnist_dataset()
            elif self.data_name == "mock_image":
                return get_mock_image_dataset()
            else:
                raise NotImplementedError

        def get_mock_image_dataset():
            batch_size = self.batch_size
            height = 1200
            width = 800
            n_channels = 3

            def gen():
                i = 0
                while True:
                    i += 1
                    yield np.random.rand(height, width, n_channels), i % 10
                    # yield np.ones((height, width, n_channels)), i % 10
                    if i > self.num_mocked_examples:
                        break

            output_types = (tf.float32, tf.float32)
            train_data = (
                tf.data.Dataset.from_generator(gen, output_types=output_types)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            val_data = tf.data.Dataset.from_generator(
                gen, output_types=output_types
            ).batch(batch_size)
            test_data = tf.data.Dataset.from_generator(
                gen, output_types=output_types
            ).batch(batch_size)
            return train_data, val_data, test_data

        def get_mnist_dataset():
            batch_size = 32
            num_val_samples = 10000

            # Return the MNIST dataset in the form of a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

            # Preprocess the data (these are Numpy arrays)
            x_train = x_train.reshape(-1, 784).astype("float32") / 255
            x_test = x_test.reshape(-1, 784).astype("float32") / 255
            y_train = y_train.astype("float32")
            y_test = y_test.astype("float32")

            # Reserve num_val_samples samples for validation
            x_val = x_train[-num_val_samples:]
            y_val = y_train[-num_val_samples:]
            x_train = x_train[:-num_val_samples]
            y_train = y_train[:-num_val_samples]
            return (
                tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
                    batch_size
                ),
                tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
                tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
            )

        if self.strategy == "mirrored":
            strategy = tf.distribute.MirroredStrategy()
            if hasattr(strategy._extended, "_collective_ops"):
                atexit.register(strategy._extended._collective_ops._pool.close)
        else:
            strategy = tf.distribute.get_strategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))

        # Open a strategy scope.
        with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
            model = get_model()

        # Train the model on all available devices.
        train_dataset, val_dataset, test_dataset = get_dataset()
        model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=val_dataset,
            # use_multiprocessing=True,
            # workers=20,
            # max_queue_size=10,
            # verbose=2,
        )
        LOGGER.info("Model training done.")

        # Test the model on all available devices.
        # model.evaluate(test_dataset, verbose=2)

    def _generate_mock_tfrecords():
        pass


if __name__ == "__main__":
    ExampleMultiGpuTrainingSession(
        data_name="mock_image",
        model_name="resnet50",
        strategy="mirrored",
        batch_size=2 * 8,
    ).run()
