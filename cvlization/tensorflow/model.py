import logging
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter


LOGGER = logging.getLogger(__name__)

_store = {}


def get_grad_accumulation_variables(model):
    if model not in _store:
        _store[model] = {
            "gradient_variables": [
                tf.Variable(
                    tf.zeros_like(v, dtype=tf.float32),
                    trainable=False,
                    name="accu_grad_" + v.name.split(":")[0],
                )
                for v in model.trainable_variables
            ],
            "gradient_zeros": [
                tf.zeros_like(trainable_var, dtype=tf.float32)
                for trainable_var in model.trainable_variables
            ],
            "n_gradients": tf.constant(model._n_gradients, dtype=tf.int32),
            "n_accum_step": tf.Variable(0, dtype=tf.int32, trainable=False),
        }
    return _store[model]


class Model(keras.Model):
    """
    A drop-in replacement for `tensorflow.keras.Model`. A custom training loop
    is implemented to support gradient accumulation.

    Warning: Please do not change the name of this class. It needs to be
    "Model", otherwise custom_objects will be needed, and even with
    custom_objects provided, keras still complains about missing inputs
    and outputs when calling the class constructor.

    When you load the model using tf.keras.models.load_model(), the returned
    object is a Functional model. The _n_gradient attribute will be lost. So
    please use the following code to set the n_gradients:

    ```
    functional_model = tf.keras.models.load_model(...)
    model = cvlization.keras.Model.from_functional_model(functional_model)
    ```

    TODO: multi-gpu training with distribution strategy does not work. when
    n_gradients > 1. Example error message:

    RuntimeError: `merge_call` called while defining a new graph or a tf.function.
    This can often happen if the function `fn` passed to `strategy.run()` contains
    a nested `@tf.function`, and the nested `@tf.function` contains a synchronization
    point, such as aggregating gradients (e.g, optimizer.apply_gradients), or if the
    function `fn` uses a control flow statement which contains a synchronization
    point in the body. Such behaviors are not yet supported. Instead, please avoid
    nested `tf.function`s or control flow statements that may potentially cross a
    synchronization boundary, for example, wrap the `fn` passed to `strategy.run`
    or the entire `strategy.run` inside a `tf.function` or move the control flow out
    of `fn`. If you are subclassing a `tf.keras.Model`, please avoid decorating
    overridden methods `test_step` and `train_step` in `tf.function`.
    """

    # TODO: consider customizing the fit() method.
    # https://github.com/keras-team/keras/blob/70d7d07bd186b929d81f7a8ceafff5d78d8bd701/keras/engine/training.py#L808
    def __init__(self, *args, n_gradients: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_gradient_accumulation(n_gradients)
        LOGGER.info(f"A custom keras model is created, with n_gradients={n_gradients}")

    @classmethod
    def from_functional_model(cls, model, n_gradients: int = 1):
        """
        Create a custom keras model from a functional model.
        """
        custom_model = cls(
            n_gradients=n_gradients, inputs=model.inputs, outputs=model.outputs
        )
        if hasattr(model, "loss"):
            custom_model.compile(
                loss=model.loss, optimizer=model.optimizer, metrics=model.metrics
            )
        return custom_model

    # @classmethod
    # def from_config(cls, config: dict):
    #     return cls(**config)

    def get_config(self):
        base_config = super().get_config()
        config = {"n_gradients": self._n_gradients}
        return dict(list(base_config.items()) + list(config.items()))

    def setup_gradient_accumulation(self, n_gradients):
        self._n_gradients = n_gradients
        if n_gradients is None:
            self._use_gradient_accumulation = False
            self.n_gradients = 1.0
        elif n_gradients <= 1:
            self._use_gradient_accumulation = False
        else:
            self._use_gradient_accumulation = True
            get_grad_accumulation_variables(self)

    @property
    def n_accum_step(self):
        return get_grad_accumulation_variables(self)["n_accum_step"]

    @property
    def gradient_accumulation(self):
        return get_grad_accumulation_variables(self)["gradient_variables"]

    @property
    def gradient_zeros(self):
        return get_grad_accumulation_variables(self)["gradient_zeros"]

    def train_step(self, data, **kwargs):
        if not self._use_gradient_accumulation:
            return super().train_step(data, **kwargs)
        self.n_accum_step.assign_add(1)

        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        if x is None or y is None:
            # Skip this step, but return current metrics.
            # What if this is the first training step?
            tf.print("A batch is skipped at step:", self.n_accum_step)
            return {m.name: m.result() for m in self.metrics}
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        if self._use_gradient_accumulation:
            # Calculate batch gradients
            gradients = tape.gradient(loss, self.trainable_variables)
            # tf.print(f"Gradient accumulation is active")
            # Accumulate batch gradients
            n_gradients = get_grad_accumulation_variables(self)["n_gradients"]
            for i in range(len(self.gradient_accumulation)):
                self.gradient_accumulation[i].assign_add(
                    gradients[i] / tf.cast(n_gradients, tf.float32)
                )

            # If n_accum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
            tf.cond(
                tf.equal(self.n_accum_step, n_gradients),
                self.apply_accu_gradients,
                lambda: None,
            )
        else:

            try:
                self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
            except TypeError:
                LOGGER.error(f"optimizer: {self.optimizer}")
                LOGGER.error(str(help(self.optimizer.minimize)))
                # For older versions of tensorflow, tape is not a valid argument. In this case, use
                # this following method instead.
                self.optimizer.apply_gradients(
                    zip(
                        tape.gradient(loss, self.trainable_variables),
                        self.trainable_variables,
                    )
                )

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        LOGGER.info("applying gradient for gradient accumulation")
        # tf.print("Applying gradient")
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.trainable_variables)
        )

        # reset
        self.n_accum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(self.gradient_zeros[i])
