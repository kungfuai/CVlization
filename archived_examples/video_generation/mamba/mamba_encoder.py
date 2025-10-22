from typing import List

import torch
from mamba_ssm import Mamba
from transformers.tokenization_utils_base import BatchEncoding

class MambaEncoder(torch.nn.Module):
    def __init__(self, n_tokens: int, n_embed: int) -> None:
        super().__init__()
        self.token_dictionary = torch.nn.Embedding(n_tokens, n_embed)
        self.sa_head = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=n_embed,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embed, n_embed),
            torch.nn.Dropout(0.2),
        )
        self.ln1 = torch.nn.LayerNorm(n_embed)
        self.ln2 = torch.nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape == (B, L)
        """
        assert len(x.shape) == 2, f"Expected (B, L), got {x.shape}"
        x = self.token_dictionary(x)

        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x # (B, L, D)
