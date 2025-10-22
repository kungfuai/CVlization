
import torch

from mamba_encoder import MambaEncoder

class MambaClassifier(torch.nn.Module):
    def __init__(self, n_tokens: int, n_embed: int, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.encoder = MambaEncoder(n_tokens, n_embed)
        self.fc = torch.nn.Linear(n_embed, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape == (B, L)
        """
        x = self.encoder(x) # (B, L, D)
        x = self.fc(x) # (B, L, C)
        assert x.shape[-1] == self.n_classes
        return x
