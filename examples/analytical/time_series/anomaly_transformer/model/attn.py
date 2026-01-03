"""
Anomaly-Attention mechanism for computing association discrepancy.
Based on ICLR 2022 paper: "Anomaly Transformer: Time Series Anomaly Detection
with Association Discrepancy"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class AnomalyAttention(nn.Module):
    """
    Anomaly-Attention mechanism that computes both:
    1. Prior-association (Gaussian kernel based on temporal distance)
    2. Series-association (learned attention weights)

    The discrepancy between these two associations is used for anomaly detection.
    """

    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # Distances for prior-association (Gaussian kernel)
        window_size = win_size
        distances = torch.zeros((window_size, window_size))
        for i in range(window_size):
            for j in range(window_size):
                distances[i][j] = abs(i - j)
        # Register as buffer so it moves with model to correct device
        self.register_buffer('distances', distances)

    def forward(self, queries, keys, values, sigma, attn_mask):
        """
        Args:
            queries: (B, L, H, E)
            keys: (B, S, H, E)
            values: (B, S, H, D)
            sigma: Temperature parameter for prior-association
            attn_mask: Attention mask

        Returns:
            values: Attention output
            series_association: Learned attention weights
            prior_association: Gaussian kernel based association
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        # Compute series-association (standard attention)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Series association (softmax of attention scores)
        series_association = self.dropout(torch.softmax(scale * scores, dim=-1))

        # Compute prior-association (Gaussian kernel)
        sigma = sigma.reshape(B, H, -1)  # B H 1
        prior_association = torch.zeros((B, L, S), device=queries.device)
        for i in range(L):
            prior_association[:, i, :] = torch.exp(-self.distances[i, :S] / sigma.reshape(B * H, -1)).reshape(B, H, S).mean(dim=1)

        # Normalize prior-association
        prior_association = prior_association / prior_association.sum(dim=-1, keepdim=True)

        # Apply series-association to values
        V = torch.einsum("bhls,bshd->blhd", series_association, values)

        if self.output_attention:
            return V.contiguous(), series_association, prior_association
        else:
            return V.contiguous(), None, None


class TriangularCausalMask():
    """Causal mask for autoregressive attention."""

    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AttentionLayer(nn.Module):
    """Attention layer wrapper with multi-head support."""

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Compute sigma before reshaping queries
        sigma = self.sigma_projection(queries).view(B, L, H)

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, series, prior = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior
