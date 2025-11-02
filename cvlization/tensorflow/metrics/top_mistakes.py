import tensorflow as tf


class TopMistakes(tf.keras.metrics.Metric):
    def __init__(self, name="top_mistakes", n=10, input_index=0, **kwargs):
        super(TopMistakes, self).__init__(name=name, **kwargs)
        self.n = n
        self.input_index = input_index
        self._false_positives = []
        self._false_negatives = []

    def update_state(self, *args, **kwargs):
        return self.update_state_with_inputs_and_outputs(*args, **kwargs)

    def update_state_with_inputs_and_outputs(
        self, y_true, y_pred, train_example=None, sample_weight=None
    ):
        if train_example is None:
            return

        # Taking the first input tensor for now.
        # TODO: generalize to multiple inputs.
        train_example = train_example[self.input_index]

        y_true = y_true.numpy()
        for i, pred in enumerate(y_pred.numpy()):
            true_label = y_true[i]
            if true_label == 1:
                self._update_top_n(pred, self._false_negatives, train_example[i], min)
            else:
                self._update_top_n(pred, self._false_positives, train_example[i], max)

    def _update_top_n(self, val, array, train_example, eval_func):
        # TODO (nice to have): use heap for faster insert + sort.
        train_example = train_example.numpy()

        if len(array) < self.n:
            array.append((val, train_example))
        else:
            to_replace = eval_func(array)

            # if the min/max val is less wrong than the current val
            if eval_func(val, to_replace[0]) != val:
                array.remove(to_replace)
                array.append((val, train_example))

    def get_inputs(self):
        return {
            "Lowest Scored Positives": self._false_positives,
            "Highest Scored Negatives": self._false_negatives,
        }

    def result(self):
        worst_fn_pred = (
            min(self._false_negatives)[0][0] if len(self._false_negatives) > 0 else 0
        )
        worst_fp_pred = (
            max(self._false_positives)[0][0] if len(self._false_positives) > 0 else 0
        )
        return worst_fp_pred + (1 - worst_fn_pred)

    def reset_state(self):
        # Using reset state means values aren't persisted after fit
        # self._false_negatives = []
        # self._false_positives = []
        pass

    def get_config(self):
        config = super().get_config()
        config.update({"n": self.n, "input_index": self.input_index})
        return config
