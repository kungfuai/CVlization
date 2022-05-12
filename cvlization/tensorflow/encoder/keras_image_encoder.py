from dataclasses import dataclass, field as dataclass_field
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import logging
from typing import List, Callable, Union


LOGGER = logging.getLogger(__name__)


@dataclass
class KerasImageEncoder:
    """
    An image encoder extracts a latent vector or tensor from an image.

    It is composed of a backbone model (a.k.a. foundation model, trunk model),
    as well as optional pooling, dense and conv layers on top of it.

    This class provides a simple way to configure add-ons to a backbone model.

    The output of an encoder shouldn't directly be the model targets.
    """

    name: str = "image_encoder"
    backbone: Union[keras.Model, Callable[..., keras.Model]] = None
    pool_name: str = "avg"
    activation: str = "relu"
    dropout: float = 0
    dense_layer_sizes: List[int] = dataclass_field(default_factory=list)
    conv_layer_sizes: List[int] = dataclass_field(default_factory=list)
    conv_kernel_size: int = 3
    use_batch_norm: bool = False
    permute_image: bool = False

    def __post_init__(self):
        self._layers_are_prepared = False

    def _prepare_layers(self):
        """
        Do not create actual layers until being called.
        """
        if not self._layers_are_prepared:
            self._prepare_dropout_layers()
            self._prepare_dense_layers()
            self._prepare_conv_layers()
            self._prepare_pooling_layers()
            self._flatten = layers.Flatten()
            self._activation = layers.Activation(self.activation)
            self._bn = layers.BatchNormalization() if self.use_batch_norm else None
            assert self.backbone is not None, "Backbone model or factory is None"
            if isinstance(self.backbone, keras.Model) or callable(self.backbone):
                # Valid type of backbone. We can hope that self.backbone can be applied as a function on
                # input tensors, to return output tensors.
                pass
            else:
                raise ValueError(
                    f"Invalid backbone model or factory: {type(self.backbone)}"
                )
            self._layers_are_prepared = True

    def __call__(self, x):
        self._prepare_layers()
        if self.permute_image:
            x = keras.layers.Permute((2, 3, 1))(x)
        x = self.backbone(x)

        if len(x.shape) == 2:
            if self.dropout > 0:
                x = self._dropout(x)
            return x

        if self.pool_name is None:
            # Pixelwise classification and then pool.
            for conv in self._convs:
                x = conv(x)
                if self.use_batch_norm:
                    x = self._bn(x)
                x = self._activation(x)
            if self.dropout > 0:
                x = self._dropout(x)
            # x is a 4D tensor.
        else:
            if self.pool_name == "avg":
                x = self._avg_pool(x)
            elif self.pool_name == "max":
                x = self._max_pool(x)
            elif self.pool_name == "flatten":
                x = self._flatten(x)
            else:
                raise NotImplementedError
            for d in self._dense_layers:
                x = d(x)
                if self.use_batch_norm:
                    x = self._bn(x)
                x = self._activation(x)
            if self.dropout > 0:
                x = self._dropout(x)
        return x

    def _layer_name(self, i):
        parts = filter(lambda x: x is not None, [self.name, str(i + 1)])
        return "_".join(parts)

    def _prepare_dense_layers(self):
        self._dense_layers = [
            layers.Dense(s, name=self._layer_name(i))
            for i, s in enumerate(self.dense_layer_sizes or [])
        ]

    def _prepare_conv_layers(self):
        self._convs = [
            layers.Conv2D(filters=s, kernel_size=self.conv_kernel_size, padding="same")
            for s in self.conv_layer_sizes
        ]

    def _prepare_dropout_layers(self):
        self._dropout = keras.layers.Dropout(rate=self.dropout)

    def _prepare_pooling_layers(self):
        if self.pool_name == "avg":
            self._avg_pool = keras.layers.GlobalAveragePooling2D()
        elif self.pool_name == "max":
            self._max_pool = keras.layers.GlobalMaxPooling2D()
