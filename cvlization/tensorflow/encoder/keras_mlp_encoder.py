from dataclasses import dataclass, field as dataclass_field
from tensorflow.keras import layers
from typing import List


@dataclass
class KerasMLPEncoder:
    # TODO: rename to tabular encoder
    dropout: float = 2
    activation: str = "relu"
    name: str = None
    dense_layer_sizes: List[int] = dataclass_field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def __post_init__(self):
        self._dense_layers = [
            layers.Dense(s, activation=self.activation, name=self._layer_name(i))
            for i, s in enumerate(self.dense_layer_sizes)
        ]

    def _layer_name(self, i):
        parts = filter(lambda x: x is not None, [self.name, str(i + 1)])
        return "_".join(parts)

    def __call__(self, x):
        for layer in self._dense_layers:
            x = layer(x)
        return x
