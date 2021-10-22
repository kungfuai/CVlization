import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.keras.engine import data_adapter


LOGGER = logging.getLogger(__name__)


class EagerModel(keras.Model):
    """A drop-in replacement for keras.Model, implemented
    in eager mode so that more flexible metrics and callbacks
    can be added.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_eagerly = True

    def train_step(self, data):
        assert tf.executing_eagerly(), "Eager execution expected."
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        if self.loss and y is None:
            raise TypeError(
                f"Target data is missing. Your model has `loss`: {self.loss}, "
                "and therefore expects target data to be passed in `fit()`."
            )

        # Compute gradients
        # trainable_vars = self.trainable_variables
        # gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        # self.compiled_metrics.update_state(y, y_pred, sample_weight)

        # Update metrics
        return_metrics = self._evaluate_metrics(y, y_pred, x)

        # Return a dict mapping metric names to current value
        return return_metrics

    def _evaluate_metrics(self, y, y_pred, x):
        if isinstance(y_pred, EagerTensor):
            y_pred = [y_pred]
        return_metrics = {}
        LOGGER.debug(f"{len(self.metrics)} metrics")
        LOGGER.debug(self.metrics)
        LOGGER.debug("compiled:", self.compiled_metrics.metrics)
        for target_idx, metrics_for_this_target in enumerate(
            self.compiled_metrics._user_metrics
        ):
            for metric in metrics_for_this_target:
                LOGGER.debug(f"To update metric: {metric} for target {target_idx}")
                LOGGER.debug(f"y: {len(y)}, {type(y)}, {y[0].shape}")
                LOGGER.debug(
                    f"y_pred: {len(y_pred)}, {type(y_pred)}, {y_pred[0].shape}"
                )
                if hasattr(metric, "update_state_with_inputs_and_outputs"):
                    metric.update_state_with_inputs_and_outputs(
                        y[target_idx], y_pred[target_idx], train_example=x
                    )
                else:
                    metric.update_state(y[target_idx], y_pred[target_idx])
        for metric in self.compiled_metrics.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics
