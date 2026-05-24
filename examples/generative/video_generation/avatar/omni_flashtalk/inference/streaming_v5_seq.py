"""Diagnostic: v5's algorithm (pre-generated noise) but on a SINGLE GPU
sequentially. No multiprocessing, no NCCL. Used to bisect v5's quality
regression — if v5_seq matches v2's output, the bug is in multi-process /
NCCL. If v5_seq matches v5 (degraded), the bug is the pre-generated noise
scheme itself.

Run with:
    python streaming_v5_seq.py --num-chunks 2
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
    ap.add_argument("--motion-latent-frames", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/home/whadmin/zz/omni_flashtalk_data/streaming_v5_seq")
    args = ap.parse_args()

    sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
    sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")
    from train_stage1_ode import (
        OmniAudioCausalWanModel, load_omni_into_causal_adapter,
        load_wan_vae, decode_latent,
        TEXT_LEN, TEXT_DIM, NUM_FRAME_PER_BLOCK,
        DENOISING_STEPS, SIGMAS)
    from PIL import Image

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    model = OmniAudioCausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=TEXT_LEN,
        in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=TEXT_DIM,
        out_dim=16, num_heads=12, num_layers=30,
        local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-6)
    model.num_frame_per_block = NUM_FRAME_PER_BLOCK
    model = load_omni_into_causal_adapter(model, args.wan_base, args.omni_lora)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    vae = load_wan_vae(args.wan_vae, device)

    lat = torch.load(os.path.join(args.data_dir, "latents", f"{args.item_id}.pt"),
                     map_location="cpu", weights_only=False)
    audio_full = lat["audio_emb"].float()
    text = lat["text_context"].float()
    ref = lat["ref_latent"].float()
    H, W = ref.shape[2], ref.shape[3]
    F_CHUNK = NUM_FRAME_PER_BLOCK
    T_AUDIO_CHUNK = 4 * F_CHUNK - 3
    seq_len = F_CHUNK * H * W // 4
    S = len(DENOISING_STEPS)

    ref_dev = ref[:, 0].unsqueeze(0).to(device, torch.bfloat16)
    text_dev = text.unsqueeze(0).to(device, torch.bfloat16)
    motion = ref[:, 0:1].repeat(1, args.motion_latent_frames, 1, 1).unsqueeze(0).to(
        device, torch.bfloat16)

    # === v5's noise pre-generation scheme (the same logic as in v5) =========
    noise_shape = (args.num_chunks, S, 1, 16, F_CHUNK, H, W)
    noise_all = torch.empty(noise_shape, device=device, dtype=torch.bfloat16)
    g = torch.Generator(device=device).manual_seed(args.seed)
    for c in range(args.num_chunks):
        for s in range(S):
            noise_all[c, s] = torch.randn(
                (1, 16, F_CHUNK, H, W),
                generator=g, device=device, dtype=torch.bfloat16)
    print(f"noise_all generated: shape={tuple(noise_all.shape)}, "
          f"mean={float(noise_all.float().mean()):.4f}, "
          f"std={float(noise_all.float().std()):.4f}")

    def run_one_step(noisy, audio_dev, step_idx):
        sigma = SIGMAS[step_idx]
        t_val = DENOISING_STEPS[step_idx]
        mask = torch.zeros(1, 1, F_CHUNK, H, W, device=device, dtype=torch.bfloat16)
        mask[:, :, args.motion_latent_frames:] = 1.0
        ref_tiled = ref_dev.unsqueeze(2).expand(-1, -1, F_CHUNK, -1, -1)
        dit_in = [torch.cat([noisy[0], ref_tiled[0], mask[0]], dim=0)]
        ctx = [text_dev[0]]
        t_frames = torch.full((1, F_CHUNK), t_val, device=device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            flow_pred = model(dit_in, t_frames, ctx, seq_len, audio_emb=audio_dev)
        x0 = noisy - sigma * flow_pred
        x0[:, :, :args.motion_latent_frames] = motion
        return x0

    print(f"\nrunning {args.num_chunks} chunks SEQUENTIALLY on one GPU with PRE-GEN noise...")
    cleans = []
    for chunk_idx in range(args.num_chunks):
        a0 = chunk_idx * F_CHUNK * 4
        audio_chunk = audio_full[a0:a0 + T_AUDIO_CHUNK]
        if audio_chunk.shape[0] < T_AUDIO_CHUNK:
            audio_chunk = F.pad(audio_chunk, (0, 0, 0, T_AUDIO_CHUNK - audio_chunk.shape[0]))
        audio_dev = audio_chunk.unsqueeze(0).to(device, torch.bfloat16)

        # initial noise from pre-gen
        noisy = noise_all[chunk_idx, 0].clone()
        noisy[:, :, :args.motion_latent_frames] = motion

        for step_idx in range(S):
            x0 = run_one_step(noisy, audio_dev, step_idx)
            if step_idx < S - 1:
                next_sigma = SIGMAS[step_idx + 1]
                # PRE-GEN noise for this transition (matches v5 indexing exactly)
                new_noise = noise_all[chunk_idx, step_idx + 1]
                noisy = (1 - next_sigma) * x0 + next_sigma * new_noise
                noisy[:, :, :args.motion_latent_frames] = motion
            else:
                cleans.append(x0)
        print(f"  chunk {chunk_idx}: clean stats mean={float(x0.float().mean()):.4f} std={float(x0.float().std()):.4f}")

    print(f"\ndecoding...")
    for ci, clean in enumerate(cleans):
        pix = decode_latent(vae, clean.squeeze(0), device)
        F_pix = pix.shape[1]
        for fl, f_idx in [("first", 0), ("mid", F_pix // 2), ("last", F_pix - 1)]:
            img = ((pix[:, f_idx] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
            Image.fromarray(img).save(
                os.path.join(args.out_dir, f"chunk{ci:02d}_{fl}.png"))
        print(f"  chunk {ci}: saved 3 PNGs")


if __name__ == "__main__":
    main()
