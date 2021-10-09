from dataclasses import dataclass
from tensorflow.keras import layers
from typing import List


@dataclass
class KerasAggregator:
    def __call__(self, x: List):
        flatten = layers.Flatten()
        flattened = [flatten(tensor) for tensor in x]
        concat = layers.Concatenate()
        concatenated = concat(flattened)
        return concatenated
