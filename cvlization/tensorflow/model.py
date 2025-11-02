import logging
import tensorflow as tf
from tensorflow import keras

try:
    from keras.utils import unpack_x_y_sample_weight  # type: ignore
except ImportError:  # pragma: no cover - fallback for Keras packaging changes
    def unpack_x_y_sample_weight(data):
        """Minimal fallback compatible with Keras' dataset outputs."""
        if isinstance(data, (list, tuple)):
            if len(data) == 3:
                return data[0], data[1], data[2]
            if len(data) == 2:
                return data[0], data[1], None
            if len(data) == 1:
                return data[0], None, None
            raise ValueError(f"Unable to unpack data with length {len(data)}")
        if isinstance(data, dict):
            x = data.get("x") or data.get("inputs")
            y = data.get("y") or data.get("targets")
            sample_weight = data.get("sample_weight")
            return x, y, sample_weight
        # Treat everything else as features only
        return data, None, None


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
        loss = getattr(model, "loss", None)
        optimizer = getattr(model, "optimizer", None)
        metrics = getattr(model, "metrics", None)
        if loss is not None:
            custom_model.compile(
                loss=loss,
                optimizer=optimizer or "adam",
                metrics=metrics,
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
        if self._use_gradient_accumulation:
            self.n_accum_step.assign_add(1)

        if isinstance(data, list):
            data = tuple(data)
        x, y, sample_weight = unpack_x_y_sample_weight(data)
        if x is None or y is None:
            # Skip this step, but return current metrics.
            # What if this is the first training step?
            tf.print("A batch is skipped at step:", self.n_accum_step)
            return {m.name: m.result() for m in self.metrics}
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(
                x,
                y,
                y_pred,
                sample_weight=sample_weight,
            )
        gradients = tape.gradient(loss, self.trainable_variables)

        if self._use_gradient_accumulation:
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
            if gradients:
                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables)
                )

        return self._finalize_train_step(x, y, y_pred, sample_weight)

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

    def _finalize_train_step(self, x, y, y_pred, sample_weight):
        self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)
        self._apply_input_aware_metrics(x, y, y_pred, sample_weight)
        return self._collect_metric_results()

    def _apply_input_aware_metrics(self, x, y, y_pred, sample_weight):
        metric_map = getattr(self, "_input_aware_metric_map", None)
        if not metric_map:
            return

        y_list = list(y) if isinstance(y, (list, tuple)) else [y]
        y_pred_list = list(y_pred) if isinstance(y_pred, (list, tuple)) else [y_pred]

        for metric in self.metrics:
            if not hasattr(metric, "update_state_with_inputs_and_outputs"):
                continue
            indices = metric_map.get(metric.name)
            if not indices:
                continue
            for idx in indices:
                target_y = y_list[idx] if idx < len(y_list) else y_list[-1]
                target_pred = (
                    y_pred_list[idx] if idx < len(y_pred_list) else y_pred_list[-1]
                )
                metric.update_state_with_inputs_and_outputs(
                    target_y,
                    target_pred,
                    train_example=x,
                    sample_weight=sample_weight,
                )

    def _collect_metric_results(self):
        results = {}
        for metric in self.metrics:
            if not hasattr(metric, "result"):
                continue
            value = metric.result()
            if isinstance(value, dict):
                results.update(value)
            else:
                results[metric.name] = value
        return results
