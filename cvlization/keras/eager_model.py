import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.keras.engine import data_adapter

from .model import Model


LOGGER = logging.getLogger(__name__)


class EagerModel(Model):
    """A drop-in replacement for keras.Model, implemented
    in eager mode so that more flexible metrics and callbacks
    can be added.
    """

    def __init__(self, *args, n_gradients: int = None, **kwargs):
        super().__init__(*args, n_gradients=n_gradients, **kwargs)
        self.run_eagerly = True

    def train_step(self, data):
        assert tf.executing_eagerly(), "Eager execution expected."
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        self.n_acum_step.assign_add(1)

        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)

        if self._use_gradient_accumulation:
            tf.print(f"Gradient accumulation is active")
            # Accumulate batch gradients
            for i in range(len(self.gradient_accumulation)):
                self.gradient_accumulation[i].assign_add(gradients[i])

            # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
            tf.cond(
                tf.equal(self.n_acum_step, self.n_gradients),
                self.apply_accu_gradients,
                lambda: None,
            )
        else:
            self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        if self.loss and y is None:
            raise TypeError(
                f"Target data is missing. Your model has `loss`: {self.loss}, "
                "and therefore expects target data to be passed in `fit()`."
            )

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
