"""D1: Pre-compute teacher (OmniAvatar 25-step, CFG=4.5, text+ref only) clean
latents for items 0..N-1. Saves to teacher_clean/<id>.pt as
{"clean_latent": [16, F_CHUNK, H, W] bf16, "ref": ref_latent, "text": text_context}.

For step distillation: the student will be trained to predict this clean
latent in a SINGLE forward pass from a forward-noised version of it.
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
    ap.add_argument("--num-items", type=int, default=10)
    ap.add_argument("--wan-base", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    ap.add_argument("--omni-lora", default="/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt")
    ap.add_argument("--wan-vae", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    ap.add_argument("--num-steps", type=int, default=25)
    ap.add_argument("--cfg-scale", type=float, default=4.5)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--motion-latent-frames", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-pngs", action="store_true",
                    help="also decode + save a mid-frame PNG for visual sanity check")
    ap.add_argument("--out-dir", default="/home/whadmin/zz/omni_flashtalk_data/distill2/teacher_clean")
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

    sched = FlowMatchScheduler(
        num_inference_steps=args.num_steps,
        shift=args.shift,
        sigma_min=0.003 / 1.002,
        extra_one_step=True,
    )
    sched.set_timesteps(num_inference_steps=args.num_steps, shift=args.shift,
                         denoising_strength=1.0)
    sigmas = sched.sigmas.tolist()
    timesteps = sched.timesteps.tolist()
    print(f"  schedule: {args.num_steps} steps, shift={args.shift}, cfg={args.cfg_scale}")
    print(f"  AUDIO DISABLED (text+ref only)")

    print("loading model...")
    model = OmniAudioCausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=TEXT_LEN,
        in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=TEXT_DIM,
        out_dim=16, num_heads=12, num_layers=30,
        local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-6)
    model.num_frame_per_block = NUM_FRAME_PER_BLOCK
    model = load_omni_into_causal_adapter(model, args.wan_base, args.omni_lora)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    vae = load_wan_vae(args.wan_vae, device) if args.save_pngs else None

    F_CHUNK = NUM_FRAME_PER_BLOCK

    def fwd(noisy, t_val, audio_d, txt, ref_dev, motion):
        H, W = noisy.shape[3], noisy.shape[4]
        seq_len = F_CHUNK * H * W // 4
        mask = torch.zeros(1, 1, F_CHUNK, H, W, device=device, dtype=torch.bfloat16)
        mask[:, :, args.motion_latent_frames:] = 1.0
        ref_tiled = ref_dev.unsqueeze(2).expand(-1, -1, F_CHUNK, -1, -1)
        dit_in = [torch.cat([noisy[0], ref_tiled[0], mask[0]], dim=0)]
        ctx = [txt[0]]
        t_frames = torch.full((1, F_CHUNK), t_val, device=device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(dit_in, t_frames, ctx, seq_len, audio_emb=audio_d)

    t_start = time.time()
    done = skipped = 0
    for item_idx in range(args.num_items):
        item_id = f"{item_idx:05d}"
        out_pt = os.path.join(args.out_dir, f"{item_id}.pt")
        if os.path.exists(out_pt):
            skipped += 1
            continue
        lat_pt = os.path.join(args.data_dir, "latents", f"{item_id}.pt")
        if not os.path.exists(lat_pt):
            print(f"  WARN {item_id}: no latents .pt; skip")
            continue
        lat = torch.load(lat_pt, map_location="cpu", weights_only=False)
        text = lat["text_context"].float()
        ref = lat["ref_latent"].float()
        H, W = ref.shape[2], ref.shape[3]

        ref_dev = ref[:, 0].unsqueeze(0).to(device, torch.bfloat16)
        text_dev = text.unsqueeze(0).to(device, torch.bfloat16)
        text_neg = torch.zeros_like(text_dev)
        motion = ref[:, 0:1].repeat(1, args.motion_latent_frames, 1, 1).unsqueeze(0).to(
            device, torch.bfloat16)

        g = torch.Generator(device=device).manual_seed(args.seed + item_idx)
        noisy = torch.randn(1, 16, F_CHUNK, H, W, generator=g,
                            device=device, dtype=torch.bfloat16)
        noisy[:, :, :args.motion_latent_frames] = motion

        for step_idx in range(args.num_steps):
            sigma = sigmas[step_idx]
            t_val = timesteps[step_idx]
            pred_pos = fwd(noisy, t_val, None, text_dev, ref_dev, motion)
            if args.cfg_scale != 1.0:
                pred_neg = fwd(noisy, t_val, None, text_neg, ref_dev, motion)
                flow_pred = pred_neg + args.cfg_scale * (pred_pos - pred_neg)
            else:
                flow_pred = pred_pos
            sigma_next = sigmas[step_idx + 1] if step_idx + 1 < len(sigmas) else 0.0
            noisy = noisy + flow_pred * (sigma_next - sigma)
            noisy[:, :, :args.motion_latent_frames] = motion

        clean = noisy.squeeze(0).cpu().to(torch.bfloat16)  # [16, F_CHUNK, H, W]
        torch.save({
            "clean_latent": clean,
            "ref_latent": lat["ref_latent"].to(torch.bfloat16),
            "text_context": lat["text_context"].to(torch.bfloat16),
            "num_steps": args.num_steps, "cfg_scale": args.cfg_scale,
            "shift": args.shift, "seed": args.seed + item_idx,
        }, out_pt)

        if args.save_pngs and vae is not None:
            pix = decode_latent(vae, clean.to(device, torch.bfloat16), device)
            img = ((pix[:, pix.shape[1]//2] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
            Image.fromarray(img).save(os.path.join(args.out_dir, f"{item_id}_mid.png"))

        done += 1
        elapsed = time.time() - t_start
        print(f"[{done}/{args.num_items}] {item_id}  std={float(clean.float().std()):.4f}  "
              f"elapsed={elapsed:.1f}s", flush=True)

    print(f"\ndone: {done} new + {skipped} skipped in {(time.time()-t_start)/60:.1f}m")


if __name__ == "__main__":
    main()
