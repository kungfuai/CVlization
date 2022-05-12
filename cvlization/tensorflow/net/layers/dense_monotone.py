import tensorflow as tf
from tensorflow.keras import layers


# TODO: use more generic names.

class CumulativeProbabilityLayer(layers.Layer):
    def __init__(self, max_followup: int):
        super(CumulativeProbabilityLayer, self).__init__()

        # Define model layers
        self._hazards_fc = layers.Dense(max_followup)
        self._base_hazard_fc = layers.Dense(max_followup)
        self._relu = layers.Activation("relu")

        # Define upper triangular mask
        mask = tf.ones([max_followup, max_followup])
        self._upper_triangular_mask = tf.linalg.band_part(mask, 0, -1)

    def call(self, inputs, **kwargs):
        raw_hazards = self._hazards_fc(inputs)
        hazards = self._relu(raw_hazards)

        expanded_hazards = tf.expand_dims(hazards, axis=-1)
        masked_hazards = tf.math.multiply(expanded_hazards, self._upper_triangular_mask)

        cum_prob = tf.math.reduce_sum(masked_hazards, axis=1) + self._base_hazard_fc(
            (inputs)
        )

        return cum_prob
