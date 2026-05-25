"""Step-distillation Phase 0: render OmniAvatar at arbitrary num_steps + cfg.

Used to verify the teacher/student gap before training:
  --num-steps 25 --cfg-scale 4.5  -> teacher reference (high quality)
  --num-steps 4  --cfg-scale 1.0  -> student baseline (low quality)

If the two visibly differ, distillation has something to learn.
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
    ap.add_argument("--num-steps", type=int, default=25,
                    help="number of denoising steps. 25 = teacher, 4 = student baseline")
    ap.add_argument("--cfg-scale", type=float, default=4.5,
                    help="classifier-free guidance scale; 1.0 = no CFG")
    ap.add_argument("--shift", type=float, default=5.0,
                    help="sigma shift; OmniAvatar uses 5.0")
    ap.add_argument("--motion-latent-frames", type=int, default=1)
    ap.add_argument("--num-chunks", type=int, default=1,
                    help="how many F_CHUNK-frame chunks to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--trained-ckpt", default=None,
                    help="path to a trained checkpoint (LoRA+audio+patch_embedding from "
                         "train_step_distill.py); overrides OmniAvatar's pretrained values")
    args = ap.parse_args()

    sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
    sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")
    from train_stage1_ode import (
        OmniAudioCausalWanModel, load_omni_into_causal_adapter,
        load_wan_vae, decode_latent,
        TEXT_LEN, TEXT_DIM, NUM_FRAME_PER_BLOCK)
    from OmniAvatar.schedulers.flow_match import FlowMatchScheduler
    from PIL import Image

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    # ---- Schedule -----------------------------------------------------------
    sched = FlowMatchScheduler(
        num_inference_steps=args.num_steps,
        shift=args.shift,
        sigma_min=0.003 / 1.002,
        extra_one_step=True,
    )
    sched.set_timesteps(num_inference_steps=args.num_steps, shift=args.shift,
                         denoising_strength=1.0)
    sigmas = sched.sigmas.tolist()      # length = num_inference_steps
    timesteps = sched.timesteps.tolist()
    print(f"  schedule: {args.num_steps} steps, shift={args.shift}")
    print(f"  sigmas (first 5 / last 5): "
          f"{[round(s,3) for s in sigmas[:5]]} ... {[round(s,3) for s in sigmas[-5:]]}")
    print(f"  cfg_scale: {args.cfg_scale}")

    # ---- Model + VAE --------------------------------------------------------
    print("loading model...")
    model = OmniAudioCausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=TEXT_LEN,
        in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=TEXT_DIM,
        out_dim=16, num_heads=12, num_layers=30,
        local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-6)
    model.num_frame_per_block = NUM_FRAME_PER_BLOCK
    model = load_omni_into_causal_adapter(model, args.wan_base, args.omni_lora)
    if args.trained_ckpt is not None:
        sd = torch.load(args.trained_ckpt, map_location="cpu", weights_only=False)
        # Filter to keys the model has, then load (trainable subset)
        model_keys = set(dict(model.named_parameters()).keys())
        usable = {k: v for k, v in sd.items() if k in model_keys}
        missing = set(sd.keys()) - model_keys
        msg = f"  trained_ckpt: loaded {len(usable)}/{len(sd)} keys"
        if missing:
            msg += f"  (skipped {len(missing)}: e.g. {next(iter(missing))})"
        print(msg)
        # Overlay the trained params onto the model
        with torch.no_grad():
            for k, v in usable.items():
                model.state_dict()[k].copy_(v.to(model.state_dict()[k].device, model.state_dict()[k].dtype))
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    vae = load_wan_vae(args.wan_vae, device)

    # ---- Inputs -------------------------------------------------------------
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
    text_neg = torch.zeros_like(text_dev)
    motion = ref[:, 0:1].repeat(1, args.motion_latent_frames, 1, 1).unsqueeze(0).to(
        device, torch.bfloat16)

    def fwd(noisy, t_val, audio_d, txt):
        mask = torch.zeros(1, 1, F_CHUNK, H, W, device=device, dtype=torch.bfloat16)
        mask[:, :, args.motion_latent_frames:] = 1.0
        ref_tiled = ref_dev.unsqueeze(2).expand(-1, -1, F_CHUNK, -1, -1)
        dit_in = [torch.cat([noisy[0], ref_tiled[0], mask[0]], dim=0)]
        ctx = [txt[0]]
        t_frames = torch.full((1, F_CHUNK), t_val, device=device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(dit_in, t_frames, ctx, seq_len, audio_emb=audio_d)

    # ---- Generate chunks ----------------------------------------------------
    g = torch.Generator(device=device).manual_seed(args.seed)
    all_clean = []
    t_start = time.time()
    for chunk_idx in range(args.num_chunks):
        a0 = chunk_idx * F_CHUNK * 4
        audio_chunk = audio_full[a0:a0 + T_AUDIO_CHUNK]
        if audio_chunk.shape[0] < T_AUDIO_CHUNK:
            audio_chunk = F.pad(audio_chunk, (0, 0, 0, T_AUDIO_CHUNK - audio_chunk.shape[0]))
        audio_dev = audio_chunk.unsqueeze(0).to(device, torch.bfloat16)
        audio_neg = torch.zeros_like(audio_dev)

        noisy = torch.randn(1, 16, F_CHUNK, H, W, generator=g,
                            device=device, dtype=torch.bfloat16)
        noisy[:, :, :args.motion_latent_frames] = motion

        # OmniAvatar's denoising loop using FlowMatchScheduler.step
        for step_idx in range(args.num_steps):
            sigma = sigmas[step_idx]
            t_val = timesteps[step_idx]
            pred_pos = fwd(noisy, t_val, audio_dev, text_dev)
            if args.cfg_scale != 1.0:
                pred_neg = fwd(noisy, t_val, audio_neg, text_neg)
                flow_pred = pred_neg + args.cfg_scale * (pred_pos - pred_neg)
            else:
                flow_pred = pred_pos
            # FlowMatchScheduler step (Euler): prev_sample = sample + model_output * (sigma_next - sigma)
            sigma_next = sigmas[step_idx + 1] if step_idx + 1 < len(sigmas) else 0.0
            noisy = noisy + flow_pred * (sigma_next - sigma)
            noisy[:, :, :args.motion_latent_frames] = motion

        all_clean.append(noisy)
        print(f"  chunk {chunk_idx+1}/{args.num_chunks}  done at {time.time()-t_start:.1f}s "
              f"clean_std={float(noisy.float().std()):.4f}")

    # ---- Decode + save ------------------------------------------------------
    print(f"\ndecoding...")
    for ci, clean in enumerate(all_clean):
        pix = decode_latent(vae, clean.squeeze(0), device)
        F_pix = pix.shape[1]
        for fl, f_idx in [("first", 0), ("mid", F_pix // 2), ("last", F_pix - 1)]:
            img = ((pix[:, f_idx] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
            Image.fromarray(img).save(
                os.path.join(args.out_dir, f"chunk{ci:02d}_{fl}.png"))
        print(f"  chunk {ci}: saved 3 PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
