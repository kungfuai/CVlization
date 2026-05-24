"""Streaming inference, Phase 1: chunked-sequential, NO KV cache.

The simplest correct sequential inference: process chunks one at a time, each
chunk = 6 latent frames (matching num_frame_per_block). For each chunk, run
the 4-step denoising loop using OmniAudioCausalWanModel's training-style
forward (full attention over the chunk). Motion-frame anchoring carries the
last few frames of the previous chunk as the visual conditioning for the
next chunk, exactly as SoulX does.

No KV cache here. Attention is recomputed per chunk over its own 6 frames.
This loses the long-video efficiency of cached attention but gives us a
fully correct sequential reference to verify against SoulX before adding
the KV cache (Phase 2) and pipeline parallelism (Phase 3).

The goal of Phase 1: confirm we can drive the trained student through full
denoising for one full chunk and decode a coherent face. Without this
working we cannot meaningfully add PP on top.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/whadmin/zz/omni_flashtalk_data")
    ap.add_argument("--item-id", default="00000")
    ap.add_argument("--wan-base", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    ap.add_argument("--omni-lora", default="/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt")
    ap.add_argument("--wan-vae", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    ap.add_argument("--num-chunks", type=int, default=1,
                    help="how many 6-latent-frame chunks to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/home/whadmin/zz/omni_flashtalk_data/streaming_v1")
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
    import numpy as np
    import math

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    # ---- Load model + VAE -------------------------------------------------
    print(f"loading student model...")
    model = OmniAudioCausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=TEXT_LEN,
        in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=TEXT_DIM,
        out_dim=16, num_heads=12, num_layers=30,
        local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-6)
    model.num_frame_per_block = NUM_FRAME_PER_BLOCK
    model = load_omni_into_causal_adapter(model, args.wan_base, args.omni_lora)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    print(f"loading Wan VAE...")
    vae = load_wan_vae(args.wan_vae, device)

    # ---- Load this item's audio + text + ref latent ----------------------
    lat_path = os.path.join(args.data_dir, "latents", f"{args.item_id}.pt")
    blob = torch.load(lat_path, map_location="cpu", weights_only=False)
    audio_full = blob["audio_emb"].float()           # [T_audio, 10752]
    text = blob["text_context"].float()              # [512, 4096]
    ref = blob["ref_latent"].float()                 # [16, 1, 96, 56]
    print(f"  audio_emb: {tuple(audio_full.shape)}")
    print(f"  text:      {tuple(text.shape)}  norm={float(text.norm()):.1f}")
    print(f"  ref:       {tuple(ref.shape)}")

    F_CHUNK = NUM_FRAME_PER_BLOCK    # 6 latent frames per chunk
    T_AUDIO_CHUNK = 4 * F_CHUNK - 3  # 21 audio frames per chunk

    H, W = ref.shape[2], ref.shape[3]
    seq_len = F_CHUNK * H * W // 4

    ref_dev = ref[:, 0].unsqueeze(0).to(device, torch.bfloat16)   # [1, 16, H, W]
    text_dev = text.unsqueeze(0).to(device, torch.bfloat16)        # [1, 512, 4096]

    # ---- Sequential chunked denoising loop --------------------------------
    print(f"\ngenerating {args.num_chunks} chunk(s) of {F_CHUNK} latent frames each...")
    all_clean = []  # list of [16, F_CHUNK, H, W] clean latents per chunk
    g = torch.Generator(device=device).manual_seed(args.seed)
    t_start = time.time()
    for chunk_idx in range(args.num_chunks):
        # Audio slice for this chunk (no overlap; in real streaming there's
        # a sliding-window with motion-frame overlap, but for the smoke we
        # take strictly non-overlapping slices).
        a0 = chunk_idx * F_CHUNK * 4
        audio_chunk = audio_full[a0:a0 + T_AUDIO_CHUNK]
        if audio_chunk.shape[0] < T_AUDIO_CHUNK:
            audio_chunk = F.pad(audio_chunk, (0, 0, 0, T_AUDIO_CHUNK - audio_chunk.shape[0]))
        audio_dev = audio_chunk.unsqueeze(0).to(device, torch.bfloat16)

        # Start from fresh noise at sigma=1.0
        noisy = torch.randn(
            1, 16, F_CHUNK, H, W,
            generator=g, device=device, dtype=torch.bfloat16)

        # 4-step denoising at SoulX's shifted timesteps
        for step_idx in range(len(DENOISING_STEPS)):
            sigma = SIGMAS[step_idx]
            t_val = DENOISING_STEPS[step_idx]

            # Build 33-channel DiT input: cat([noisy, ref_tiled, mask])
            mask = torch.zeros(1, 1, F_CHUNK, H, W, device=device, dtype=torch.bfloat16)
            mask[:, :, 1:] = 1.0
            ref_tiled = ref_dev.unsqueeze(2).expand(-1, -1, F_CHUNK, -1, -1)
            dit_in = [torch.cat([noisy[0], ref_tiled[0], mask[0]], dim=0)]
            ctx = [text_dev[0]]
            t_frames = torch.full((1, F_CHUNK), t_val, device=device, dtype=torch.bfloat16)

            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                flow_pred = model(dit_in, t_frames, ctx, seq_len, audio_emb=audio_dev)
            # convert flow_pred -> x0_pred
            x0_pred = noisy - sigma * flow_pred

            # Re-noise to next timestep (unless last step)
            if step_idx < len(DENOISING_STEPS) - 1:
                next_sigma = SIGMAS[step_idx + 1]
                new_noise = torch.randn_like(noisy)
                noisy = (1 - next_sigma) * x0_pred + next_sigma * new_noise
            else:
                clean = x0_pred  # final clean latent

        all_clean.append(clean.squeeze(0))   # [16, F_CHUNK, H, W]
        elapsed = time.time() - t_start
        print(f"  chunk {chunk_idx+1}/{args.num_chunks}  done at {elapsed:.1f}s "
              f"(avg {elapsed/(chunk_idx+1):.2f}s/chunk)")

    # ---- Decode and save --------------------------------------------------
    print(f"\ndecoding...")
    for ci, clean in enumerate(all_clean):
        pix = decode_latent(vae, clean, device)  # [3, F_pix, H, W]
        F_pix = pix.shape[1]
        # save first / middle / last frame as PNGs for inspection
        for fl, f_idx in [("first", 0), ("mid", F_pix // 2), ("last", F_pix - 1)]:
            img = ((pix[:, f_idx] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
            Image.fromarray(img).save(
                os.path.join(args.out_dir, f"chunk{ci:02d}_{fl}.png"))
        print(f"  chunk {ci}: pix {tuple(pix.shape)} -> 3 PNGs in {args.out_dir}")


if __name__ == "__main__":
    main()
