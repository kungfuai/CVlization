import logging
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from keras.utils import unpack_x_y_sample_weight

from .model import Model


LOGGER = logging.getLogger(__name__)


class EagerModel(Model):
    """A drop-in replacement for keras.Model, implemented
    in eager mode so that more flexible metrics and callbacks
    can be added.
    """

    def __init__(self, *args, n_gradients: int = 1, **kwargs):
        super().__init__(*args, n_gradients=n_gradients, **kwargs)
        self.run_eagerly = True

    def train_step(self, data):
        assert tf.executing_eagerly(), "Eager execution expected."
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if self._use_gradient_accumulation:
            self.n_accum_step.assign_add(1)

        if isinstance(data, list):
            data = tuple(data)
        x, y, sample_weight = unpack_x_y_sample_weight(data)
        if x is None or y is None:
            # Skip this step, but return current metrics.
            # What if this is the first training step?
            tf.print("A batch is skipped at step:", self.n_accum_step)
            # TODO: is self.metrics enough? Does it include the custom metrics like TopMistakes?
            return {m.name: m.result() for m in self.metrics}

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compute_loss(
                x,
                y,
                y_pred,
                sample_weight=sample_weight,
            )

        gradients = tape.gradient(loss, self.trainable_variables)

        if self._use_gradient_accumulation:
            # tf.print("Gradient accumulation is active")
            # Accumulate batch gradients
            for i in range(len(self.gradient_accumulation)):
                self.gradient_accumulation[i].assign_add(
                    gradients[i] / tf.cast(self._n_gradients, tf.float32)
                )

            # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
            tf.cond(
                tf.equal(self.n_accum_step, self._n_gradients),
                self.apply_accu_gradients,
                lambda: None,
            )
        else:
            if gradients:
                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables)
                )

        if self.loss and y is None:
            raise TypeError(
                f"Target data is missing. Your model has `loss`: {self.loss}, "
                "and therefore expects target data to be passed in `fit()`."
            )

        return self._finalize_train_step(x, y, y_pred, sample_weight)
