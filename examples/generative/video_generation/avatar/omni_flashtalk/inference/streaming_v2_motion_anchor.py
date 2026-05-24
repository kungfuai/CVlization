"""Streaming inference, Phase 2: chunked-sequential with motion-frame anchoring.

Adds two things on top of v1:

1. **Motion-frame clamping**: at every denoising step, the first
   `motion_latent_frames` of the noisy latent are overwritten with the
   VAE-encoded ref image (chunk 0) or the last N clean latent frames of the
   previous chunk (chunk 1+). This is SoulX's i2v anchoring trick — without
   it the model has nothing to ground the face on and the first frames of
   each chunk decode to noise.

2. **Cross-chunk continuity**: after each chunk, the last `motion_latent_frames`
   clean latent frames become the next chunk's anchor.

Still no KV cache — attention is recomputed per chunk over its own 6 frames.

Verification target: chunk 0 frame 0 should decode to the ref image (the
woman with wine glasses), and chunk 1 should be visually continuous with
chunk 0.
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
    ap.add_argument("--num-chunks", type=int, default=2)
    ap.add_argument("--motion-latent-frames", type=int, default=1,
                    help="how many leading latent frames to clamp as motion anchor "
                         "(SoulX's motion_frames_num=5 pixel frames => ~1 latent frame)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/home/whadmin/zz/omni_flashtalk_data/streaming_v2")
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

    print(f"loading Wan VAE...")
    vae = load_wan_vae(args.wan_vae, device)

    # ---- Load this item's audio + text + ref latent ----------------------
    lat_path = os.path.join(args.data_dir, "latents", f"{args.item_id}.pt")
    blob = torch.load(lat_path, map_location="cpu", weights_only=False)
    audio_full = blob["audio_emb"].float()
    text = blob["text_context"].float()
    ref = blob["ref_latent"].float()                 # [16, 1, H, W]
    H, W = ref.shape[2], ref.shape[3]
    print(f"  audio_emb: {tuple(audio_full.shape)}")
    print(f"  ref:       {tuple(ref.shape)}")

    F_CHUNK = NUM_FRAME_PER_BLOCK
    T_AUDIO_CHUNK = 4 * F_CHUNK - 3
    seq_len = F_CHUNK * H * W // 4

    ref_dev = ref[:, 0].unsqueeze(0).to(device, torch.bfloat16)         # [1, 16, H, W]
    text_dev = text.unsqueeze(0).to(device, torch.bfloat16)             # [1, 512, 4096]

    # The motion-anchor is the encoded ref latent, tiled over the first N
    # latent frames. For chunk 0 we use the ref. For chunk N+1 we use the
    # last N latent frames of chunk N's clean output.
    motion = ref[:, 0:1].repeat(1, args.motion_latent_frames, 1, 1).unsqueeze(0).to(
        device, torch.bfloat16)   # [1, 16, motion_lat_frames, H, W]

    print(f"\ngenerating {args.num_chunks} chunk(s) of {F_CHUNK} latent frames each, "
          f"motion_latent_frames={args.motion_latent_frames}...")

    all_clean = []
    g = torch.Generator(device=device).manual_seed(args.seed)
    t_start = time.time()
    for chunk_idx in range(args.num_chunks):
        # Audio slice for this chunk
        a0 = chunk_idx * F_CHUNK * 4
        audio_chunk = audio_full[a0:a0 + T_AUDIO_CHUNK]
        if audio_chunk.shape[0] < T_AUDIO_CHUNK:
            audio_chunk = F.pad(audio_chunk, (0, 0, 0, T_AUDIO_CHUNK - audio_chunk.shape[0]))
        audio_dev = audio_chunk.unsqueeze(0).to(device, torch.bfloat16)

        # Start from fresh noise at sigma=1.0, but CLAMP motion frames
        noisy = torch.randn(
            1, 16, F_CHUNK, H, W,
            generator=g, device=device, dtype=torch.bfloat16)
        noisy[:, :, :args.motion_latent_frames] = motion   # i2v anchor

        for step_idx in range(len(DENOISING_STEPS)):
            sigma = SIGMAS[step_idx]
            t_val = DENOISING_STEPS[step_idx]

            mask = torch.zeros(1, 1, F_CHUNK, H, W, device=device, dtype=torch.bfloat16)
            mask[:, :, args.motion_latent_frames:] = 1.0    # mask AFTER motion frames
            ref_tiled = ref_dev.unsqueeze(2).expand(-1, -1, F_CHUNK, -1, -1)
            dit_in = [torch.cat([noisy[0], ref_tiled[0], mask[0]], dim=0)]
            ctx = [text_dev[0]]
            t_frames = torch.full((1, F_CHUNK), t_val, device=device, dtype=torch.bfloat16)

            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                flow_pred = model(dit_in, t_frames, ctx, seq_len, audio_emb=audio_dev)
            x0_pred = noisy - sigma * flow_pred
            # Force motion frames to stay = anchor (the model shouldn't change them)
            x0_pred[:, :, :args.motion_latent_frames] = motion

            if step_idx < len(DENOISING_STEPS) - 1:
                next_sigma = SIGMAS[step_idx + 1]
                new_noise = torch.randn_like(noisy)
                noisy = (1 - next_sigma) * x0_pred + next_sigma * new_noise
                noisy[:, :, :args.motion_latent_frames] = motion
            else:
                clean = x0_pred

        all_clean.append(clean.squeeze(0))
        # carry the last `motion_latent_frames` clean frames into the next chunk
        motion = clean[:, :, -args.motion_latent_frames:].clone()
        elapsed = time.time() - t_start
        print(f"  chunk {chunk_idx+1}/{args.num_chunks}  done at {elapsed:.1f}s "
              f"(avg {elapsed/(chunk_idx+1):.2f}s/chunk)")

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
