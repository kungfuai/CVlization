from tensorflow import keras
import tensorflow as tf


class LazyModel(keras.Model):
    def __init__(self, *args, n_gradients: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        if n_gradients is None:
            self._use_gradient_accumulation = False
        else:
            self._use_gradient_accumulation = True
            self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
            self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
            self.gradient_accumulation = [
                tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
                for v in self.trainable_variables
            ]

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        if self._use_gradient_accumulation:
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
            # self.optimizer.apply_gradients(
            #     zip(gradients, self.trainable_variables)
            # )
            self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.trainable_variables)
        )

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(self.trainable_variables[i], dtype=tf.float32)
            )
