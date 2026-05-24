"""Streaming inference with CFG (classifier-free guidance).

Adds CFG on top of v2's working chunked-sequential generation. OmniAvatar's
official inference uses CFG with cfg_scale ~= 4.0 — without it, the
student's predicted velocity is weak and decoded frames come out blurry.

Per step:
  pred_pos = model(noisy, text, audio, ...)        # conditional
  pred_neg = model(noisy, neg_text, neg_audio, ...) # unconditional
  pred     = pred_neg + cfg_scale * (pred_pos - pred_neg)
  x0       = noisy - sigma * pred

This doubles per-step compute (two model calls). For real-time deployment
we'd want a CFG-distilled model (one forward pass with CFG baked in).
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
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--motion-latent-frames", type=int, default=1)
    ap.add_argument("--cfg-scale", type=float, default=4.0,
                    help="OmniAvatar uses 4.0 by default; 1.0 = no CFG (= v2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/home/whadmin/zz/omni_flashtalk_data/streaming_v6_cfg")
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

    ref_dev = ref[:, 0].unsqueeze(0).to(device, torch.bfloat16)
    text_dev = text.unsqueeze(0).to(device, torch.bfloat16)
    # Unconditional ("negative") text = zeros
    text_neg = torch.zeros_like(text_dev)
    motion = ref[:, 0:1].repeat(1, args.motion_latent_frames, 1, 1).unsqueeze(0).to(
        device, torch.bfloat16)

    def fwd(noisy, sigma, t_val, audio_dev, txt):
        mask = torch.zeros(1, 1, F_CHUNK, H, W, device=device, dtype=torch.bfloat16)
        mask[:, :, args.motion_latent_frames:] = 1.0
        ref_tiled = ref_dev.unsqueeze(2).expand(-1, -1, F_CHUNK, -1, -1)
        dit_in = [torch.cat([noisy[0], ref_tiled[0], mask[0]], dim=0)]
        ctx = [txt[0]]
        t_frames = torch.full((1, F_CHUNK), t_val, device=device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(dit_in, t_frames, ctx, seq_len, audio_emb=audio_dev)

    g = torch.Generator(device=device).manual_seed(args.seed)
    cleans = []
    t_start = time.time()
    for chunk_idx in range(args.num_chunks):
        a0 = chunk_idx * F_CHUNK * 4
        audio_chunk = audio_full[a0:a0 + T_AUDIO_CHUNK]
        if audio_chunk.shape[0] < T_AUDIO_CHUNK:
            audio_chunk = F.pad(audio_chunk, (0, 0, 0, T_AUDIO_CHUNK - audio_chunk.shape[0]))
        audio_dev = audio_chunk.unsqueeze(0).to(device, torch.bfloat16)
        # Unconditional audio = zeros (OmniAvatar's nega_audio approach)
        audio_neg = torch.zeros_like(audio_dev)

        noisy = torch.randn(1, 16, F_CHUNK, H, W, generator=g,
                            device=device, dtype=torch.bfloat16)
        noisy[:, :, :args.motion_latent_frames] = motion

        for step_idx in range(len(DENOISING_STEPS)):
            sigma = SIGMAS[step_idx]
            t_val = DENOISING_STEPS[step_idx]
            pred_pos = fwd(noisy, sigma, t_val, audio_dev, text_dev)
            if args.cfg_scale != 1.0:
                pred_neg = fwd(noisy, sigma, t_val, audio_neg, text_neg)
                flow_pred = pred_neg + args.cfg_scale * (pred_pos - pred_neg)
            else:
                flow_pred = pred_pos
            x0 = noisy - sigma * flow_pred
            x0[:, :, :args.motion_latent_frames] = motion
            print(f"  chunk {chunk_idx} step {step_idx}: x0 std={float(x0.float().std()):.4f}", flush=True)
            if step_idx < len(DENOISING_STEPS) - 1:
                next_sigma = SIGMAS[step_idx + 1]
                noisy = (1 - next_sigma) * x0 + next_sigma * torch.randn_like(noisy)
                noisy[:, :, :args.motion_latent_frames] = motion
            else:
                cleans.append(x0)
        print(f"  chunk {chunk_idx} done at {time.time()-t_start:.1f}s", flush=True)

    print(f"\ndecoding...")
    for ci, clean in enumerate(cleans):
        pix = decode_latent(vae, clean.squeeze(0), device)
        F_pix = pix.shape[1]
        for fl, f_idx in [("first", 0), ("mid", F_pix // 2), ("last", F_pix - 1)]:
            img = ((pix[:, f_idx] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
            Image.fromarray(img).save(
                os.path.join(args.out_dir, f"chunk{ci:02d}_{fl}.png"))
        print(f"  chunk {ci}: saved")


if __name__ == "__main__":
    main()
