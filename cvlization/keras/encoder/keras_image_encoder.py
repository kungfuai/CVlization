from dataclasses import dataclass, field as dataclass_field
from tensorflow import keras
from tensorflow.keras import layers
from typing import List

from ..net.lenet import LeNet


@dataclass
class KerasImageEncoder:
    """
    An image encoder extracts a latent vector or tensor from an image.

    It is composed of a foundation model (a.k.a. trunk model), as well as
    optional pooling, dense and conv layers on top of it.

    This class provides a simple way to configure add-ons to a trunk model.

    The output of an encoder shouldn't directly be the model targets.
    """

    trunk: keras.Model = LeNet()  # MobileNetV2(include_top=False, pooling=None)
    pool_name: str = "avg"
    dropout: float = 0
    # TODO: dense_layer_sizes is used for both dense layers and conv layers.
    dense_layer_sizes: List[int] = dataclass_field(default_factory=list)

    def __post_init__(self):
        self._prepare_dropout_layers()
        self._prepare_dense_layers()
        self._prepare_pooling_layers()

    def __call__(self, x):
        x = self.trunk(x)

        if self.pool_name is None:
            # Pixelwise classification and then pool.
            for conv in self._convs:
                x = conv(x)
            x = self._dropout(x)
        else:
            if self.pool_name == "avg":
                x = self._avg_pool(x)
            elif self.pool_name == "max":
                x = self._max_pool(x)
            else:
                raise NotImplementedError
            for l in self._dense_layers:
                x = l(x)
            x = self._dropout(x)
        return x

    def _prepare_dense_layers(self):
        self._dense_layers = [
            layers.Dense(s, activation=self.activation, name=self._layer_name(i))
            for i, s in enumerate(self.dense_layer_sizes)
        ]

    def _prepare_dropout_layers(self):
        self._dropout = keras.layers.Dropout(rate=self.dropout)

    def _prepare_pooling_layers(self):
        if self.pool_name is None:
            self._convs = [
                layers.Conv2D(
                    filters=s,
                    kernel_size=(3, 3),
                    padding="same",
                    activation="relu",
                )
                for s in self.dense_layer_sizes
            ]
        elif self.pool_name == "avg":
            self._avg_pool = keras.layers.GlobalAveragePooling2D()
        elif self.pool_name == "max":
            self._max_pool = keras.layers.GlobalMaxPooling2D()
