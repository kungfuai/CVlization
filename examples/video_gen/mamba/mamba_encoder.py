
import torch
from mamba_ssm import Mamba

class MambaBlock(torch.nn.Module):
    def __init__(
            self,
            mamba_n_embed: int,
            mamba_d_state: int,
            mamba_d_conv: int,
            mamba_expand: int,
    ) -> None:
        super().__init__()
        self.sa_head = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=mamba_n_embed,  # Model dimension d_model
            d_state=mamba_d_state,  # SSM state expansion factor
            d_conv=mamba_d_conv,  # Local convolution width
            expand=mamba_expand,  # Block expansion factor
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(mamba_n_embed, 4 * mamba_n_embed),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * mamba_n_embed, mamba_n_embed),
            torch.nn.Dropout(0.2),
        )
        self.ln1 = torch.nn.LayerNorm(mamba_n_embed)
        self.ln2 = torch.nn.LayerNorm(mamba_n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MambaEncoder(torch.nn.Module):
    def __init__(
            self,
            n_tokens: int,
            seq_len: int,
            mamba_n_embed: int,
            mamba_d_state: int,
            mamba_d_conv: int,
            mamba_expand: int,
            n_mamba_layers: int,
            device: str,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.seq_len = seq_len
        self.device = device
        self.token_dictionary = torch.nn.Embedding(n_tokens, mamba_n_embed)
        self.position_embedding_table = torch.nn.Embedding(seq_len, mamba_n_embed)
        self.mamba_blocks = torch.nn.Sequential(*[
            MambaBlock(
                mamba_n_embed=mamba_n_embed,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
            )
            for _ in range(n_mamba_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape == (B, L)
        """
        assert x.max() < self.n_tokens, \
            f"Max token found ({x.max()}) is >= Num tokens ({self.n_tokens})"

        assert (len(x.shape) == 2) and (x.shape[-1] == self.seq_len), \
            f"Expected (B, {self.seq_len}), got {x.shape}"

        pos_emb = self.position_embedding_table(
            torch.arange(self.seq_len, device=self.device),
        )

        x = self.token_dictionary(x) + pos_emb

        x = self.mamba_blocks(x)

        return x # (B, L, D)
