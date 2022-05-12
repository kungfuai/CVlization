from dataclasses import dataclass
import enum
import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Union


class AggregationMethod(enum.Enum):
    """
    Enum of aggregation methods.
    """

    CONCAT = "concat"
    CLASSIFY_AND_AVG = "classify_and_avg"
    CLASSIFY_AND_MAX = "classify_and_max"
    CLASSFIY_AND_CONCAT = "classify_and_concat"
    CONTRAST_AVG_CONCAT = "contrast_avg_concat"


class ImageFeaturePoolingMethod(enum.Enum):
    """
    Enum of image feature pooling methods.
    """

    AVG = "avg"
    MAX = "max"
    FLATTEN = "flatten"


@dataclass
class KerasAggregator:
    name: str = "aggregator"
    aggregation_method: Union[AggregationMethod, str] = AggregationMethod.CONCAT
    image_feature_pooling_method: Union[
        ImageFeaturePoolingMethod, str
    ] = ImageFeaturePoolingMethod.AVG
    activation: str = "relu"
    dense_layer_sizes: List[int] = None
    use_batch_norm: bool = False
    dropout: float = 0

    def __post_init__(self):
        if isinstance(self.aggregation_method, str):
            self.aggregation_method = AggregationMethod(self.aggregation_method)
        if isinstance(self.image_feature_pooling_method, str):
            self.image_feature_pooling_method = ImageFeaturePoolingMethod(
                self.image_feature_pooling_method
            )
        self._layers_are_prepared = False

    def _prepare_layers(self):
        if not self._layers_are_prepared:
            self._prepare_dense_layers()
            self._flatten = layers.Flatten()
            self._concat = layers.Concatenate()
            self._activation = layers.Activation(self.activation)
            self._bn = (
                layers.BatchNormalization(axis=-1) if self.use_batch_norm else None
            )
            self._dropout = layers.Dropout(self.dropout)
            self._classify = layers.Dense(1)

            if self.image_feature_pooling_method == ImageFeaturePoolingMethod.AVG:
                self._global_pool = layers.GlobalAveragePooling2D()
            elif self.image_feature_pooling_method == ImageFeaturePoolingMethod.MAX:
                self._global_pool = layers.GlobalMaxPooling2D()
            elif self.image_feature_pooling_method == ImageFeaturePoolingMethod.FLATTEN:
                self._global_pool = layers.Flatten()
            self._layers_are_prepared = True

    def __call__(self, x: List):
        self._prepare_layers()
        if self.aggregation_method == AggregationMethod.CONCAT:
            return self._concat_and_dense(x)
        elif self.aggregation_method == AggregationMethod.CLASSIFY_AND_AVG:
            return self._classify_and_avg(x)
        elif self.aggregation_method == AggregationMethod.CLASSIFY_AND_MAX:
            return self._classify_and_max(x)
        elif self.aggregation_method == AggregationMethod.CLASSFIY_AND_CONCAT:
            return self._classify_and_concat(x)
        raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")

    def _pool_or_flatten(self, x):
        if len(x.shape) == 4:
            # This is for an image feature tensor.
            x = self._global_pool(x)
            assert len(x.shape) == 2
            return x
        else:
            if len(x.shape) == 2:
                return x
            return self._flatten(x)

    def _concat_and_dense(self, x: List):
        flattened = [self._pool_or_flatten(tensor) for tensor in x]
        if len(flattened) == 1:
            return flattened[0]
        concatenated = self._concat(flattened)
        x = concatenated
        for d in self._dense_layers:
            x = d(x)
            if self.use_batch_norm:
                x = self._bn(x)
            x = self._activation(x)
        if len(self.dense_layer_sizes or []) > 0:
            x = self._dropout(x)
        return x

    def _classify_and_concat(self, x: List):
        # This does not add anything vs. _concat_and_dense.
        flattened = [self._pool_or_flatten(tensor) for tensor in x]
        logits = []
        for x in flattened:
            for d in self._dense_layers:
                x = d(x)
                if self.use_batch_norm:
                    x = self._bn(x)
                x = self._activation(x)
            logits.append(self._classify(x))
        return self._concat(logits)

    def _classify_and_avg(self, x: List):
        # This does not add anything vs. _concat_and_dense.
        logits = self._classify_and_concat(x)
        return tf.reduce_mean(logits, axis=-1, keepdims=True)

    def _classify_and_max(self, x: List):
        logits = self._classify_and_concat(x)
        return tf.reduce_max(logits, axis=-1, keepdims=True)

    def _prepare_dense_layers(self):
        # TODO: this logic is the same as in KerasImageEncoder. Abstract it out.
        self._dense_layers = [
            layers.Dense(s) for _, s in enumerate(self.dense_layer_sizes or [])
        ]
