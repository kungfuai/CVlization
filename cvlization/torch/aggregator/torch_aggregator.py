from dataclasses import dataclass
from typing import List
from torch import nn
import torch
import logging


LOGGER = logging.getLogger(__name__)


class TorchAggregator(nn.Module):
    def __init__(
        self,
        name: str = "aggregator",
        activation: str = "ReLU",
        dense_layer_sizes: List[int] = None,
        flatten_start_dim: int = 1,
        use_batch_norm: bool = False,
        dropout: float = 0,
    ):
        super().__init__()
        self.name = name
        self.activation = activation
        self.dense_layer_sizes = dense_layer_sizes
        self.flatten_start_dim = flatten_start_dim
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.__post_init__()

    def __post_init__(self):
        self._flatten = nn.Flatten(start_dim=self.flatten_start_dim)
        self._activation = getattr(torch.nn, self.activation)()
        self._dropout = nn.Dropout(self.dropout)
        self._prepare_dense_layers()

    def _ensure_enough_dim(self, tensor):
        while len(tensor.shape) <= self.flatten_start_dim:
            tensor = torch.unsqueeze(tensor, -1)
        return tensor

    def forward(self, x: List):
        x = [self._ensure_enough_dim(tensor) for tensor in x]
        for tensor in x:
            LOGGER.debug(f"To flatten {tensor.shape}")
        flattened = [self._flatten(tensor) for tensor in x]
        concatenated = torch.cat(flattened, dim=self.flatten_start_dim)
        x = concatenated
        for i, d in enumerate(self._dense_layers):
            x = d(x)
            if self.use_batch_norm:
                x = self._batch_norm_layers[i](x)
            x = self._activation(x)
        if len(self.dense_layer_sizes or []) > 0:
            x = self._dropout(x)
        return x

    def _prepare_dense_layers(self):
        _dense_layers = []
        _batch_norm_layers = []
        for out_features in self.dense_layer_sizes or []:
            _dense = nn.LazyLinear(out_features)
            _dense_layers.append(_dense)
            _batch_norm_layers.append(nn.BatchNorm1d(num_features=out_features))
        self._dense_layers = nn.ModuleList(_dense_layers)
        self._batch_norm_layers = nn.ModuleList(_batch_norm_layers)
