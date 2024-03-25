
import torch

from .mamba_encoder import MambaEncoder

class MambaClassifier(torch.nn.Module):
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
        self.n_classes = n_tokens
        self.encoder = MambaEncoder(
            n_tokens=n_tokens,
            seq_len=seq_len,
            mamba_n_embed=mamba_n_embed,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            n_mamba_layers=n_mamba_layers,
            device=device,
        )
        self.dropout = torch.nn.Dropout(0.25)
        self.norm = torch.nn.LayerNorm(mamba_n_embed)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(mamba_n_embed, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape == (B, L)
        """
        x = self.encoder(x) # (B, L, D)
        x = self.dropout(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.fc(x) # (B, L, C)
        assert x.shape[-1] == self.n_classes, \
            f"Expected {self.n_classes} classes, got {x.shape[-1]}"
        return x
