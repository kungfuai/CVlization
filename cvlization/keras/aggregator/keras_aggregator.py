from dataclasses import dataclass
from tensorflow.keras import layers
from typing import List


@dataclass
class KerasAggregator:
    name: str = "aggregator"
    activation: str = "relu"
    dense_layer_sizes: List[int] = None

    def __post_init__(self):
        self._prepare_dense_layers()

    def __call__(self, x: List):
        flatten = layers.Flatten()
        flattened = [flatten(tensor) for tensor in x]
        if len(flattened) == 1:
            return flattened[0]
        concat = layers.Concatenate()
        concatenated = concat(flattened)

        x = concatenated
        for d in self._dense_layers:
            x = d(x)
        return x

    def _prepare_dense_layers(self):
        # TODO: this logic is the same as in KerasImageEncoder. Abstract it out.
        self._dense_layers = [
            layers.Dense(s, activation=self.activation, name=self._layer_name(i))
            for i, s in enumerate(self.dense_layer_sizes or [])
        ]

    def _layer_name(self, i):
        parts = filter(lambda x: x is not None, [self.name, str(i + 1)])
        return "_".join(parts)
