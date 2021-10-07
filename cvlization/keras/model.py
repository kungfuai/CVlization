import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop


class Model(keras.Model):
    def train_step(self, data):
        # assert tf.executing_eagerly()
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        return_metrics = self._evaluate_metrics()

        # Return a dict mapping metric names to current value
        return return_metrics

    def _evaluate_metrics(self, y, y_pred, x):
        for metric in self.metrics:
            if hasattr(metric, "update_state_with_inputs_and_outputs"):
                metric.update_state_with_inputs_and_outputs(y, y_pred, train_example=x)
            elif isinstance(metric, tf.keras.metrics.Metric):
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
