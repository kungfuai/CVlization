from dataclasses import dataclass, field as dataclass_field
from typing import List, Optional
import torch
from torch import nn


class TorchImageEncoder(nn.Module):
    """
    An image encoder extracts a latent vector or tensor from an image.

    It is composed of a backbone model (a.k.a. foundation model, or trunk model),
    as well as optional pooling, dense and conv layers on top of it.

    This class provides a simple way to configure add-ons to a backbone model.

    The output of an encoder shouldn't directly be the model targets.
    """

    def __init__(
        self,
        name: str = "image_encoder",
        backbone: Optional[nn.Module] = None,
        pool_name: str = "avg",
        activation: str = "ReLU",
        image_channels: int = 3,
        dropout: float = 0,
        dense_layer_sizes: List[int] = None,
        conv_layer_sizes: List[int] = None,
        conv_kernel_size: int = 3,
        use_batch_norm: bool = True,
        permute_image: bool = True,
        customize_conv1: bool = False,
        finetune_backbone: bool = False,
    ):
        """Initialize image encoder.

        Args:
            name: Encoder name for identification
            backbone: Image backbone model (e.g., from create_image_backbone)
            pool_name: Pooling method - "avg", "max", or "flatten"
            activation: Activation function name (e.g., "ReLU", "GELU")
            image_channels: Number of input image channels
            dropout: Dropout probability
            dense_layer_sizes: Optional list of dense layer output sizes
            conv_layer_sizes: Optional list of conv layer output sizes
            conv_kernel_size: Kernel size for conv layers
            use_batch_norm: Whether to use batch normalization
            permute_image: Whether to permute image from HWC to CHW format
            customize_conv1: Whether to customize first conv layer
            finetune_backbone: Whether to unfreeze backbone for fine-tuning (default: False = frozen)
        """
        super().__init__()
        self.name = name
        self.backbone = backbone
        self.pool_name = pool_name
        self.activation = activation
        self.image_channels = image_channels
        self.dropout = dropout
        self.dense_layer_sizes = dense_layer_sizes
        self.conv_layer_sizes = conv_layer_sizes
        self.conv_kernel_size = conv_kernel_size
        self.use_batch_norm = use_batch_norm
        self.permute_image = permute_image
        self.customize_conv1 = customize_conv1
        self.finetune_backbone = finetune_backbone
        self.__post_init__()

    def __post_init__(self):
        assert self.backbone is not None, "Need an image backbone model."

        # Freeze or unfreeze backbone based on finetune_backbone flag
        if not self.finetune_backbone:
            # Freeze backbone for feature extraction only
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Unfreeze backbone for fine-tuning
            for param in self.backbone.parameters():
                param.requires_grad = True

        self._prepare_dropout_layers()
        self._prepare_dense_layers()
        self._prepare_conv_layers()
        self._prepare_pooling_layers()

        if self.customize_conv1:
            self.backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            self.backbone.maxpool = nn.Identity()

        self._flatten = nn.Flatten(start_dim=1)
        self._activation = getattr(torch.nn, self.activation)()

    def forward(self, x):
        if self.permute_image:
            x = x.permute(0, 3, 1, 2)
        x = self.backbone(x)

        if self.pool_name is None:
            # Pixelwise classification and then pool.
            for i, conv in enumerate(self._convs):
                x = conv(x)
                if self.use_batch_norm:
                    x = self._batch_norm_layers[i](x)
                x = self._activation(x)
        else:
            if self.pool_name == "avg":
                x = self._avg_pool(x)
            elif self.pool_name == "max":
                x = self._max_pool(x)
            elif self.pool_name == "flatten":
                x = self._flatten(x)
            else:
                raise NotImplementedError
            x = self._flatten(x)
            for d, batch_norm in zip(self._dense_layers, self._batch_norm_layers):
                x = d(x)
                if self.use_batch_norm:
                    x = batch_norm(x)
                x = self._activation(x)

        x = self._dropout(x)
        return x

    def _layer_name(self, i):
        parts = filter(lambda x: x is not None, [self.name, str(i + 1)])
        return "_".join(parts)

    def _prepare_dense_layers(self):
        self._dense_layers = []
        self._batch_norm_layers = []
        for out_channels in self.dense_layer_sizes or []:
            _dense = nn.LazyLinear(out_channels)
            self._dense_layers.append(_dense)
            self._batch_norm_layers.append(nn.BatchNorm1d(num_features=out_channels))
            self._dense_layers = nn.ModuleList(self._dense_layers)
            self._batch_norm_layers = nn.ModuleList(self._batch_norm_layers)

    def _prepare_dropout_layers(self):
        self._dropout = nn.Dropout(self.dropout)

    def _prepare_conv_layers(self):
        _batch_norm_layers = []
        if self.pool_name is None:
            _convs = []
            for out_channels in self.conv_layer_sizes or []:
                _conv = nn.LazyConv2d(
                    out_channels,
                    kernel_size=self.conv_kernel_size,
                    padding="same",
                )
                _convs.append(_conv)
                _batch_norm_layers.append(nn.BatchNorm2d(num_features=out_channels))
            self._convs = nn.ModuleList(_convs)
            self._batch_norm_layers = nn.ModuleList(_batch_norm_layers)

    def _prepare_pooling_layers(self):
        if self.pool_name == "avg":
            self._avg_pool = nn.AdaptiveAvgPool2d(output_size=[1, 1])
        elif self.pool_name == "max":
            self._max_pool = nn.AdaptiveMaxPool2d(output_size=[1, 1])
        elif self.pool_name == "flatten":
            self._flatten = nn.Flatten()


if __name__ == "__main__":
    from .torch_image_backbone import create_image_backbone

    t = create_image_backbone("resnet18")
    e = TorchImageEncoder(backbone=t)
    print(type(e.backbone))
