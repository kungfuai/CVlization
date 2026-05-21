"""Audio-conditioning adapter (E3b + gradient checkpointing)."""
import sys
sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")

import torch
import torch.nn as nn
from wan.modules.model import WanModel as SFWanModel
from wan.modules.model import sinusoidal_embedding_1d
from OmniAvatar.models.audio_pack import AudioPack


class OmniAudioWanModel(SFWanModel):
    """SF WanModel + OmniAvatar audio path."""

    def __init__(self, in_dim=33, audio_hidden=32, audio_input_dim=10752, **kw):
        super().__init__(in_dim=in_dim, **kw)
        dim = kw.get("dim", 1536)
        num_layers = kw.get("num_layers", 30)
        self.num_audio_layers = num_layers // 2 - 1
        self.audio_first_layer = 2
        self.audio_proj = AudioPack(audio_input_dim, [4, 1, 1], audio_hidden, layernorm=True)
        self.audio_cond_projs = nn.ModuleList(
            [nn.Linear(audio_hidden, dim) for _ in range(self.num_audio_layers)]
        )
        for l in self.audio_cond_projs:
            nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)

    def _prepare_audio(self, audio_emb, x_shape):
        audio = audio_emb.permute(0, 2, 1)[:, :, :, None, None]
        audio = torch.cat([audio[:, :, :1].repeat(1, 1, 3, 1, 1), audio], dim=2)
        audio = self.audio_proj(audio)
        audio_layers = torch.cat([p(audio) for p in self.audio_cond_projs], dim=0)
        B = audio_emb.shape[0]
        N_aud = audio_layers.shape[0] // B
        audio_layers = audio_layers.reshape(B, N_aud, *audio_layers.shape[1:])
        return audio_layers

    def _inject_audio_at_layer(self, x, audio_layers, layer_i, grid_size):
        if not (self.audio_first_layer <= layer_i < self.audio_first_layer + self.num_audio_layers):
            return x
        au_idx = layer_i - self.audio_first_layer
        F_p, H_p, W_p = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
        a = audio_layers[:, au_idx]
        if a.shape[1] != F_p:
            raise ValueError(f"audio T_aud_packed ({a.shape[1]}) != video F_patch ({F_p})")
        a = a.repeat(1, 1, H_p, W_p, 1)
        a = a.reshape(a.shape[0], F_p * H_p * W_p, -1)
        return x + a

    def _forward(self, x, t, context, seq_len, audio_emb=None, **kw):
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x_emb = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x_emb]
        )
        x_emb = [u.flatten(2).transpose(1, 2) for u in x_emb]
        seq_lens = torch.tensor([u.size(1) for u in x_emb], dtype=torch.long)
        x_emb = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x_emb
        ])

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).type_as(x_emb))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        audio_layers = None
        if audio_emb is not None:
            audio_layers = self._prepare_audio(audio_emb, x_emb.shape)

        kwargs = dict(
            e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
            freqs=self.freqs, context=context, context_lens=None,
        )
        use_ckpt = torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False)

        for layer_i, block in enumerate(self.blocks):
            if audio_layers is not None:
                x_emb = self._inject_audio_at_layer(
                    x_emb, audio_layers, layer_i, grid_sizes[0]
                )
            if use_ckpt:
                x_emb = torch.utils.checkpoint.checkpoint(
                    lambda x_, kw=kwargs: block(x_, **kw),
                    x_emb, use_reentrant=False,
                )
            else:
                x_emb = block(x_emb, **kwargs)

        x_emb = self.head(x_emb, e)
        return x_emb


from omni_adapter import load_omni_into_adapter  # noqa: E402,F401
