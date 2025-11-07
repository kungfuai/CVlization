"""
Anomaly Transformer model for time series anomaly detection.
Simplified implementation based on ICLR 2022 paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attn import AnomalyAttention, AttentionLayer


class EncoderLayer(nn.Module):
    """Transformer encoder layer with Anomaly-Attention."""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask


class Encoder(nn.Module):
    """Transformer encoder with multiple layers."""

    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list = []
        prior_list = []

        for attn_layer in self.attn_layers:
            x, series, prior = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list


class AnomalyTransformer(nn.Module):
    """
    Anomaly Transformer for multivariate time series anomaly detection.

    Key components:
    1. Token embedding + positional encoding
    2. Multi-layer encoder with Anomaly-Attention
    3. Reconstruction head
    4. Association discrepancy computation

    Args:
        win_size: Window size (sequence length)
        enc_in: Number of input features
        c_out: Number of output features (usually same as enc_in)
        d_model: Model dimension
        n_heads: Number of attention heads
        e_layers: Number of encoder layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        output_attention: Whether to output attention weights
    """

    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3,
                 d_ff=512, dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Embedding
        self.embedding = nn.Linear(enc_in, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, win_size, d_model))

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout,
                                       output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Reconstruction head
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        """
        Args:
            x: (batch_size, win_size, n_features)

        Returns:
            reconstruction: (batch_size, win_size, n_features)
            series_list: List of series-association matrices
            prior_list: List of prior-association matrices
        """
        # Embedding
        x = self.embedding(x)  # (B, L, D)

        # Add positional encoding
        x = x + self.pos_encoding

        # Encoder
        enc_out, series, prior = self.encoder(x, attn_mask=None)

        # Reconstruction
        reconstruction = self.projection(enc_out)

        if self.output_attention:
            return reconstruction, series, prior
        else:
            return reconstruction, None, None
