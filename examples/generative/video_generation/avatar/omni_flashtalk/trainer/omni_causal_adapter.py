"""Audio-conditioning adapter on top of Self-Forcing's CausalWanModel.

E4 deliverable. Same audio path as omni_adapter_v2.py, but built on the
causal model (block-causal mask + KV cache support). Requires torch >= 2.5
because CausalWanModel uses `flex_attention`.
"""
import sys
sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")

import torch
import torch.nn as nn
from wan.modules.causal_model import CausalWanModel
from wan.modules.causal_model import sinusoidal_embedding_1d
from OmniAvatar.models.audio_pack import AudioPack


class OmniAudioCausalWanModel(CausalWanModel):
    """CausalWanModel + OmniAvatar audio path.

    Audio injection follows the same OmniAvatar pattern as the bidirectional
    adapter: per-block additive condition before each of layers [2..N/2].
    Causal-specific concerns (KV cache, block_mask, local_attn_size, sink_size)
    are inherited from CausalWanModel.
    """

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

    def _prepare_audio(self, audio_emb):
        """OmniAvatar audio preprocessing -> (B, N_audio_layers, T_aud_packed, 1, 1, dim)."""
        audio = audio_emb.permute(0, 2, 1)[:, :, :, None, None]
        audio = torch.cat([audio[:, :, :1].repeat(1, 1, 3, 1, 1), audio], dim=2)
        audio = self.audio_proj(audio)
        audio_layers = torch.cat([p(audio) for p in self.audio_cond_projs], dim=0)
        B = audio_emb.shape[0]
        N_aud = audio_layers.shape[0] // B
        return audio_layers.reshape(B, N_aud, *audio_layers.shape[1:])

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

    def _forward_train(self, x, t, context, seq_len, audio_emb=None,
                       clean_x=None, aug_t=None, clip_fea=None, y=None):
        """Adapted from CausalWanModel._forward_train with per-block audio injection."""
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Causal mask
        if self.block_mask is None:
            x0_shape = x[0].shape if isinstance(x, list) else x.shape
            num_frames = x0_shape[1] if isinstance(x, list) else x.shape[2]
            H, W = (x0_shape[2], x0_shape[3]) if isinstance(x, list) else (x.shape[-2], x.shape[-1])
            frame_seqlen = H * W // (self.patch_size[1] * self.patch_size[2])
            if clean_x is not None:
                self.block_mask = self._prepare_teacher_forcing_mask(
                    device, num_frames=num_frames, frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block
                )
            else:
                self.block_mask = self._prepare_blockwise_causal_attn_mask(
                    device, num_frames=num_frames, frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size
                )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]
            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1)
                for u in clean_x
            ])
            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x)
            )
            e0_clean = self.time_projection(e_clean).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        audio_layers = None
        if audio_emb is not None:
            audio_layers = self._prepare_audio(audio_emb)

        kwargs = dict(
            e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
            freqs=self.freqs, context=context, context_lens=context_lens,
            block_mask=self.block_mask,
        )
        use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing

        for layer_i, block in enumerate(self.blocks):
            if audio_layers is not None:
                x = self._inject_audio_at_layer(x, audio_layers, layer_i, grid_sizes[0])
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    lambda x_, kw=kwargs: block(x_, **kw),
                    x, use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)


def load_omni_into_causal_adapter(model, wan_base_safetensors, omni_lora_pt,
                                   lora_rank=128, lora_alpha=128):
    from safetensors.torch import load_file
    from peft import LoraConfig, inject_adapter_in_model
    base_sd = load_file(wan_base_safetensors)
    base_sd = {k: v for k, v in base_sd.items() if not k.startswith("patch_embedding")}
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    print(f"  Wan base: {len(base_sd)} loaded, missing={len(missing)}, unexpected={len(unexpected)}")
    lora_targets = ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
                    "ffn.0", "ffn.2"]
    lora_cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=lora_targets,
                          bias="none", init_lora_weights=True)
    model = inject_adapter_in_model(lora_cfg, model)
    omni_sd = torch.load(omni_lora_pt, map_location="cpu", weights_only=False)
    missing2, unexpected2 = model.load_state_dict(omni_sd, strict=False)
    matched = len(omni_sd) - len(unexpected2)
    print(f"  OmniAvatar: {len(omni_sd)} keys, matched={matched}, unexpected={len(unexpected2)}")
    if unexpected2:
        print(f"    first 5 unexpected: {unexpected2[:5]}")
    return model
