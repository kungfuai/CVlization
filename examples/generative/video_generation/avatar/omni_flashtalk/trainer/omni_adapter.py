"""Audio-conditioning adapter on top of Self-Forcing's bidirectional WanModel.

Adds OmniAvatar's audio path (AudioPack + audio_cond_projs) and adjusts
patch_embedding's in_channels (16 -> 33) so that OmniAvatar-1.3B's released
checkpoint loads cleanly.

This module is the "bidirectional" half of the port. The causal half (KV
cache, block-causal mask, future_audio_frames) bolts on later once we set up
an env with torch >= 2.5 (Self-Forcing's CausalWanModel needs flex_attention).
"""
import sys, os
sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")

import torch
import torch.nn as nn
from einops import rearrange

from wan.modules.model import WanModel as SFWanModel
from OmniAvatar.models.audio_pack import AudioPack  # reuse upstream unchanged


class OmniAudioWanModel(SFWanModel):
    """SF WanModel + OmniAvatar audio path. Parameter naming matches OmniAvatar.

    Shapes (1.3B):
      patch_embedding.weight: (1536, 33, 1, 2, 2)   (vs SF base 16 in-channels)
      audio_proj.proj.weight: (32, 43008)
      audio_proj.norm_out.{weight,bias}: (32,)
      audio_cond_projs.{0..13}.weight: (1536, 32)
      audio_cond_projs.{0..13}.bias:   (1536,)
    """

    def __init__(self, in_dim=33, audio_hidden=32, audio_input_dim=10752, **kw):
        # Force SF base init with in_dim=33 so patch_embedding shape matches
        # OmniAvatar's checkpoint exactly.
        super().__init__(in_dim=in_dim, **kw)
        dim = kw.get("dim", 1536)
        num_layers = kw.get("num_layers", 30)

        # Audio path, lifted from OmniAvatar/models/wan_video_dit.py L311-318
        self.audio_proj = AudioPack(audio_input_dim, [4, 1, 1], audio_hidden, layernorm=True)
        self.audio_cond_projs = nn.ModuleList(
            [nn.Linear(audio_hidden, dim) for _ in range(num_layers // 2 - 1)]
        )
        # Initialize audio projections to zero so adding them is a no-op for
        # baseline parity tests (the OmniAvatar checkpoint will overwrite anyway).
        for l in self.audio_cond_projs:
            nn.init.zeros_(l.weight)
            nn.init.zeros_(l.bias)


def load_omni_into_adapter(model: OmniAudioWanModel,
                            wan_base_safetensors: str,
                            omni_lora_pt: str,
                            lora_rank: int = 128, lora_alpha: int = 128):
    """Load Wan2.1-T2V-1.3B base + OmniAvatar overlay into the adapter model.

    Steps:
      1. Load Wan base safetensors (skip patch_embedding because shape mismatch).
      2. Inject LoRA into base layers via peft (same targets OmniAvatar trained).
      3. Load OmniAvatar pytorch_model.pt - replaces patch_embedding, fills LoRA + audio.
    """
    from safetensors.torch import load_file
    from peft import LoraConfig, inject_adapter_in_model

    # Step 1: Wan base
    base_sd = load_file(wan_base_safetensors)
    # Skip patch_embedding: OmniAvatar has 33-channel version that replaces it.
    base_sd = {k: v for k, v in base_sd.items() if not k.startswith("patch_embedding")}
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    print(f"  Wan base load: {len(base_sd)} keys, missing={len(missing)} (audio+patch_embedding), unexpected={len(unexpected)}")

    # Step 2: Inject LoRA on the same targets OmniAvatar trained.
    # Targets verified from key dump: self_attn/cross_attn/ffn.{0,2} {q,k,v,o}
    lora_targets = ["q", "k", "v", "o", "0", "2"]  # 0,2 = ffn.0, ffn.2
    lora_cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha,
        target_modules=lora_targets,
        bias="none",
        init_lora_weights=True,
    )
    model = inject_adapter_in_model(lora_cfg, model)

    # Step 3: OmniAvatar overlay
    omni_sd = torch.load(omni_lora_pt, map_location="cpu", weights_only=False)
    # OmniAvatar SD uses "blocks.X.self_attn.q.lora_A.default.weight" style.
    # peft's default adapter name is also "default", so keys should match.
    # But peft inserts modules under a wrapper; we need to load via model.load_state_dict.
    missing2, unexpected2 = model.load_state_dict(omni_sd, strict=False)
    matched = len(omni_sd) - len(unexpected2)
    print(f"  OmniAvatar load: {len(omni_sd)} keys, matched={matched}, missing={len(missing2)}, unexpected={len(unexpected2)}")
    if unexpected2:
        print(f"    first 5 unexpected: {unexpected2[:5]}")
    if missing2:
        # Filter to only the actually-load-affected ones
        real_missing = [m for m in missing2 if not ("lora_" in m or "base_layer" in m)]
        print(f"    real missing (non-lora-scaffold): {len(real_missing)} (first 5: {real_missing[:5]})")

    return model


def main():
    # Construct the model. Parameters match Wan2.1-T2V-1.3B.
    model = OmniAudioWanModel(
        model_type='t2v', patch_size=(1, 2, 2), text_len=512,
        in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
        num_heads=12, num_layers=30, window_size=(-1, -1),
        qk_norm=True, cross_attn_norm=True, eps=1e-6,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Adapter model: {n_params/1e9:.2f}B params before LoRA injection")

    # Verify the new audio path is registered
    audio_keys = [k for k, _ in model.named_parameters() if "audio" in k]
    print(f"  audio params: {len(audio_keys)} (expect 32: audio_proj 4 + audio_cond_projs 14*2)")
    print(f"  patch_embedding: {tuple(model.patch_embedding.weight.shape)}")

    print()
    model = load_omni_into_adapter(
        model,
        "/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt",
    )

    # Quick sanity check: a few audio params should be non-zero now
    print()
    a0_w = model.get_parameter("audio_cond_projs.0.weight")
    print(f"  audio_cond_projs.0.weight stats: mean={a0_w.mean():.4e}  std={a0_w.std():.4e}  zeros={(a0_w==0).sum().item()}/{a0_w.numel()}")
    pe = model.patch_embedding.weight
    print(f"  patch_embedding stats: mean={pe.mean():.4e}  std={pe.std():.4e}  zeros={(pe==0).sum().item()}/{pe.numel()}")


if __name__ == "__main__":
    main()
