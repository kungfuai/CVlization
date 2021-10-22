"""
Courtesy of:
https://stackoverflow.com/questions/63580476/how-to-compute-auc-for-one-class-in-multi-class-classification-in-keras
"""
import tensorflow as tf


class MulticlassAUC(tf.keras.metrics.AUC):
    """AUC for a single class in a muliticlass problem.

    Parameters
    ----------
    pos_label : int
        Label of the positive class (the one whose AUC is being computed).

    from_logits : bool, optional (default: False)
        If True, assume predictions are not standardized to be between 0 and 1.
        In this case, predictions will be squeezed into probabilities using the
        softmax function.

    sparse : bool, optional (default: True)
        If True, ground truth labels should be encoded as integer indices in the
        range [0, n_classes-1]. Otherwise, ground truth labels should be one-hot
        encoded indicator vectors (with a 1 in the true label position and 0
        elsewhere).

    **kwargs : keyword arguments
        Keyword arguments for tf.keras.metrics.AUC.__init__(). For example, the
        curve type (curve='ROC' or curve='PR').
    """

    def __init__(self, pos_label, from_logits=False, sparse=True, **kwargs):
        super().__init__(**kwargs)

        self.pos_label = pos_label
        self.from_logits = from_logits
        self.sparse = sparse

    def update_state(self, y_true, y_pred, **kwargs):
        """Accumulates confusion matrix statistics.

        Parameters
        ----------
        y_true : tf.Tensor
            The ground truth values. Either an integer tensor of shape
            (n_examples,) (if sparse=True) or a one-hot tensor of shape
            (n_examples, n_classes) (if sparse=False).

        y_pred : tf.Tensor
            The predicted values, a tensor of shape (n_examples, n_classes).

        **kwargs : keyword arguments
            Extra keyword arguments for tf.keras.metrics.AUC.update_state
            (e.g., sample_weight).
        """
        if self.sparse:
            y_true = tf.math.equal(y_true, self.pos_label)
            y_true = tf.squeeze(y_true)
        else:
            y_true = y_true[..., self.pos_label]

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = y_pred[..., self.pos_label]

        super().update_state(y_true, y_pred, **kwargs)
