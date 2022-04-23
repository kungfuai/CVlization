from dataclasses import dataclass
from typing import List
from torch import nn
import torch


class TorchMlpEncoder(nn.Module):
    def __init__(
        self,
        name: str = "mlp_encoder",
        activation: str = "ReLU",
        dense_layer_sizes: List[int] = None,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.name = name
        self.activation = activation
        self.dense_layer_sizes = dense_layer_sizes
        self.use_batch_norm = use_batch_norm
        self._activation = getattr(torch.nn, self.activation)
        self._prepare_dense_layers()

    def forward(self, x):
        for i, d in enumerate(self._dense_layers):
            x = d(x)
            if self.use_batch_norm:
                x = self._batch_norm_layers[i](x)
            x = self._activation(x)
        return x

    def _prepare_dense_layers(self):
        _dense_layers = []
        _batch_norm_layers = []
        for out_features in self.dense_layer_sizes or []:
            _dense = nn.LazyLinear(out_features)
            _dense_layers.append(_dense)
            _batch_norm_layers.append(nn.BatchNorm1d(num_features=out_features))
        self._batch_norm_layers = nn.ModuleList(_batch_norm_layers)
        self._dense_layers = nn.ModuleList(_dense_layers)
