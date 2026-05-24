"""Streaming inference, Phase 3: chunked-sequential WITH KV cache.

Builds on v2 (which already has motion-frame anchoring + cross-chunk
continuity) by routing every model call through the adapter's
_forward_inference path, which uses the per-layer KV cache + crossattn
cache. This is the prerequisite for v4 (pipeline parallelism), because PP
requires that each denoising step be a self-contained KV-cached call.

KV cache semantics (Algorithm 2 of CausVid):
- Per chunk, run 4 denoising steps with kv_cache=current cache state
- After last step, RE-RUN with timestep=0 (clean context) to UPDATE the
  KV cache with the clean prediction
- Next chunk's calls see the clean cache state from all prior chunks

Verification target: match v2's output frames within numerical tolerance.
If the KV cache + clean-context update is correct, the visual output for
chunks 0..N should look essentially identical to v2 (modulo floating-point
noise from running with cached attention vs full attention).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F


def init_kv_cache(num_blocks, kv_cache_size, batch_size, dtype, device,
                  num_heads=12, head_dim=128):
    return [{
        "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim],
                         dtype=dtype, device=device),
        "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim],
                         dtype=dtype, device=device),
        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "local_end_index":  torch.tensor([0], dtype=torch.long, device=device),
    } for _ in range(num_blocks)]


def init_crossattn_cache(num_blocks, batch_size, dtype, device,
                          text_len=512, num_heads=12, head_dim=128):
    return [{
        "k": torch.zeros([batch_size, text_len, num_heads, head_dim],
                         dtype=dtype, device=device),
        "v": torch.zeros([batch_size, text_len, num_heads, head_dim],
                         dtype=dtype, device=device),
        "is_init": False,
    } for _ in range(num_blocks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/whadmin/zz/omni_flashtalk_data")
    ap.add_argument("--item-id", default="00000")
    ap.add_argument("--wan-base", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    ap.add_argument("--omni-lora", default="/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt")
    ap.add_argument("--wan-vae", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    ap.add_argument("--num-chunks", type=int, default=3)
    ap.add_argument("--motion-latent-frames", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/home/whadmin/zz/omni_flashtalk_data/streaming_v3")
    ap.add_argument("--no-audio", action="store_true",
                    help="diagnostic: bypass audio path entirely")
    args = ap.parse_args()

    sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
    sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")
    from train_stage1_ode import (
        OmniAudioCausalWanModel, load_omni_into_causal_adapter,
        load_wan_vae, decode_latent,
        TEXT_LEN, TEXT_DIM, NUM_FRAME_PER_BLOCK,
        DENOISING_STEPS, SIGMAS,
    )
    from PIL import Image

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    print(f"loading student model...")
    model = OmniAudioCausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=TEXT_LEN,
        in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=TEXT_DIM,
        out_dim=16, num_heads=12, num_layers=30,
        local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-6)
    model.num_frame_per_block = NUM_FRAME_PER_BLOCK
    model = load_omni_into_causal_adapter(model, args.wan_base, args.omni_lora)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    NUM_BLOCKS = len(model.blocks)

    print(f"loading Wan VAE...")
    vae = load_wan_vae(args.wan_vae, device)

    # ---- Load item ---------------------------------------------------------
    lat = torch.load(os.path.join(args.data_dir, "latents", f"{args.item_id}.pt"),
                     map_location="cpu", weights_only=False)
    audio_full = lat["audio_emb"].float()
    text = lat["text_context"].float()
    ref = lat["ref_latent"].float()
    H, W = ref.shape[2], ref.shape[3]
    F_CHUNK = NUM_FRAME_PER_BLOCK
    T_AUDIO_CHUNK = 4 * F_CHUNK - 3
    seq_len = F_CHUNK * H * W // 4
    frame_seq_length = H * W // 4   # tokens per latent frame after patchify

    ref_dev = ref[:, 0].unsqueeze(0).to(device, torch.bfloat16)
    text_dev = text.unsqueeze(0).to(device, torch.bfloat16)
    motion = ref[:, 0:1].repeat(1, args.motion_latent_frames, 1, 1).unsqueeze(0).to(
        device, torch.bfloat16)

    # ---- Init KV caches ----------------------------------------------------
    kv_cache = init_kv_cache(NUM_BLOCKS, kv_cache_size=32760, batch_size=1,
                              dtype=torch.bfloat16, device=device)
    ca_cache = init_crossattn_cache(NUM_BLOCKS, batch_size=1,
                                     dtype=torch.bfloat16, device=device)

    def call_model(noisy, t_val, audio_dev, current_start):
        """One model call via _forward_inference (KV-cached path)."""
        mask = torch.zeros(1, 1, F_CHUNK, H, W, device=device, dtype=torch.bfloat16)
        mask[:, :, args.motion_latent_frames:] = 1.0
        ref_tiled = ref_dev.unsqueeze(2).expand(-1, -1, F_CHUNK, -1, -1)
        dit_in = [torch.cat([noisy[0], ref_tiled[0], mask[0]], dim=0)]
        ctx = [text_dev[0]]
        t_frames = torch.full((1, F_CHUNK), t_val, device=device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(dit_in, t_frames, ctx, seq_len,
                         audio_emb=None if args.no_audio else audio_dev,
                         kv_cache=kv_cache, crossattn_cache=ca_cache,
                         current_start=current_start)

    # ---- Streaming loop ----------------------------------------------------
    print(f"\ngenerating {args.num_chunks} chunk(s) with KV cache...")
    all_clean = []
    g = torch.Generator(device=device).manual_seed(args.seed)
    current_start = 0   # token offset in global sequence
    t_start = time.time()
    for chunk_idx in range(args.num_chunks):
        a0 = chunk_idx * F_CHUNK * 4
        audio_chunk = audio_full[a0:a0 + T_AUDIO_CHUNK]
        if audio_chunk.shape[0] < T_AUDIO_CHUNK:
            audio_chunk = F.pad(audio_chunk, (0, 0, 0, T_AUDIO_CHUNK - audio_chunk.shape[0]))
        audio_dev = audio_chunk.unsqueeze(0).to(device, torch.bfloat16)

        noisy = torch.randn(1, 16, F_CHUNK, H, W, generator=g,
                            device=device, dtype=torch.bfloat16)
        noisy[:, :, :args.motion_latent_frames] = motion

        # 4-step denoising — all calls share the SAME current_start since
        # they all process the same chunk
        for step_idx in range(len(DENOISING_STEPS)):
            sigma = SIGMAS[step_idx]
            t_val = DENOISING_STEPS[step_idx]
            flow_pred = call_model(noisy, t_val, audio_dev, current_start)
            x0_pred = noisy - sigma * flow_pred
            x0_pred[:, :, :args.motion_latent_frames] = motion
            if step_idx < len(DENOISING_STEPS) - 1:
                next_sigma = SIGMAS[step_idx + 1]
                new_noise = torch.randn_like(noisy)
                noisy = (1 - next_sigma) * x0_pred + next_sigma * new_noise
                noisy[:, :, :args.motion_latent_frames] = motion
            else:
                clean = x0_pred

        # Update KV cache with CLEAN context (timestep=0) — populates the
        # cache so the next chunk attends to the CLEAN prediction of this one
        _ = call_model(clean, 0.0, audio_dev, current_start)

        all_clean.append(clean.squeeze(0))
        motion = clean[:, :, -args.motion_latent_frames:].clone()
        current_start += F_CHUNK * frame_seq_length

        elapsed = time.time() - t_start
        print(f"  chunk {chunk_idx+1}/{args.num_chunks}  done at {elapsed:.1f}s "
              f"(avg {elapsed/(chunk_idx+1):.2f}s/chunk)  "
              f"kv_cache global_end={kv_cache[0]['global_end_index'].item()}")

    # ---- Decode -----------------------------------------------------------
    print(f"\ndecoding...")
    for ci, clean in enumerate(all_clean):
        pix = decode_latent(vae, clean, device)
        F_pix = pix.shape[1]
        for fl, f_idx in [("first", 0), ("mid", F_pix // 2), ("last", F_pix - 1)]:
            img = ((pix[:, f_idx] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
            Image.fromarray(img).save(
                os.path.join(args.out_dir, f"chunk{ci:02d}_{fl}.png"))
        print(f"  chunk {ci}: pix {tuple(pix.shape)} -> 3 PNGs in {args.out_dir}")


if __name__ == "__main__":
    main()
