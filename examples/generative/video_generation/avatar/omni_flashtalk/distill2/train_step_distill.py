"""D2: Step-distillation trainer.

Teacher = OmniAvatar 1.3B at 25 steps + CFG=4.5 (pre-computed clean latents).
Student = OmniAvatar 1.3B + LoRA (trainable), targeting 4-step inference.

Per training step:
  1. sample (item, t_idx) where t_idx in {0,1,2,3} = the 4-step student schedule
  2. forward-noise teacher's clean to get xt at sigma_t
  3. student does ONE forward pass at sigma_t -> predict velocity
  4. pred_x0 = xt - sigma_t * student_velocity
  5. loss = MSE(pred_x0, teacher_clean)
  6. backward, update LoRA + audio + patch_embedding

This is the textbook consistency-distillation objective. Student learns
to skip ahead from any sigma_t directly to clean x0 in one step.

Why this WILL show loss decreasing where Stage-1 ODE training didn't:
  Stage-1 used SoulX teacher targets -- student already produced similar
  latents (loss = 0.057, near a local min). Here, the student at 4-step
  produces blurry mush (std~0.33) while teacher gives sharp output
  (std~0.9). The gap is real and learnable.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from omni_causal_adapter import OmniAudioCausalWanModel, load_omni_into_causal_adapter

# 4-step student schedule (matches our v6/v7 setup)
DENOISING_STEPS_RAW = [1000.0, 750.0, 500.0, 250.0]
TIMESTEP_SHIFT = 5.0
NUM_TIMESTEPS = 1000.0
def _shift(t):
    f = t / NUM_TIMESTEPS
    return NUM_TIMESTEPS * (TIMESTEP_SHIFT * f / (1 + (TIMESTEP_SHIFT - 1) * f))
DENOISING_STEPS = [_shift(t) for t in DENOISING_STEPS_RAW]
SIGMAS = [t / NUM_TIMESTEPS for t in DENOISING_STEPS]

NUM_FRAME_PER_BLOCK = 6
TEXT_LEN, TEXT_DIM = 512, 4096


class TeacherDataset(Dataset):
    def __init__(self, teacher_dir):
        self.files = sorted(glob.glob(os.path.join(teacher_dir, "*.pt")))
        print(f"  TeacherDataset: {len(self.files)} items")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        return {
            "clean_latent": b["clean_latent"].float(),   # [16, F, H, W]
            "ref_latent":   b["ref_latent"].float(),     # [16, 1, H, W]
            "text_context": b["text_context"].float(),   # [text_len, text_dim]
        }


def make_dit_input(xt, ref, mask):
    ref_tiled = ref.unsqueeze(1).expand(-1, xt.shape[1], -1, -1)
    return torch.cat([xt, ref_tiled, mask], dim=0)


def compute_loss(model, batch, t_idx, motion_latent_frames=1, noise_seed=None,
                  loss_weighting="uniform"):
    """Step-distillation loss for one item + one timestep choice.
    loss_weighting:
      uniform  - MSE(pred_x0, target)
      sigma_sq - MSE * sigma_t**2  (amplifies high-sigma which are hardest)
      inv_sig  - MSE / max(sigma_t, 0.1) (de-amplifies high-sigma; focuses on easy)
      min_snr  - EDM-style; MSE * min(SNR, 1) where SNR = (1-sigma)**2 / sigma**2
    """
    device = "cuda"
    dtype = torch.bfloat16
    clean = batch["clean_latent"].to(device, dtype)         # [B, 16, F, H, W]
    ref   = batch["ref_latent"][:, :, 0].to(device, dtype)  # [B, 16, H, W]
    text  = batch["text_context"].to(device, dtype)         # [B, text_len, text_dim]
    B, _, Fc, H, W = clean.shape
    motion = clean[:, :, :motion_latent_frames]              # anchor = first latent frame(s) of teacher's clean

    sigma = SIGMAS[t_idx]
    t_val = DENOISING_STEPS[t_idx]

    # Forward-noise teacher's clean to get xt at sigma
    if noise_seed is not None:
        g = torch.Generator(device=device).manual_seed(noise_seed)
        noise = torch.randn(clean.shape, generator=g, device=device, dtype=dtype)
    else:
        noise = torch.randn_like(clean)
    xt = (1 - sigma) * clean + sigma * noise
    # Re-clamp motion frames to the clean teacher motion
    xt[:, :, :motion_latent_frames] = motion

    # Build 33-channel DiT input
    mask = torch.zeros(B, 1, Fc, H, W, device=device, dtype=dtype)
    mask[:, :, motion_latent_frames:] = 1.0
    dit_in = [make_dit_input(xt[i], ref[i], mask[i]) for i in range(B)]
    ctx = [text[i] for i in range(B)]
    t_frames = torch.full((B, Fc), t_val, device=device, dtype=dtype)
    seq_len = Fc * H * W // 4

    flow_pred = model(dit_in, t_frames, ctx, seq_len, audio_emb=None)
    pred_x0 = xt - sigma * flow_pred
    # Don't penalize motion-frame mismatch (they're trivially clamped to teacher's)
    # Loss over non-motion frames only
    pred_eval = pred_x0[:, :, motion_latent_frames:]
    target_eval = clean[:, :, motion_latent_frames:]
    base = F.mse_loss(pred_eval.float(), target_eval.float())
    if loss_weighting == "sigma_sq":
        weight = sigma ** 2
    elif loss_weighting == "inv_sig":
        weight = 1.0 / max(sigma, 0.1)
    elif loss_weighting == "min_snr":
        # SNR = (1-sigma)**2 / sigma**2; cap at 1.0 a la EDM Min-SNR
        snr = ((1 - sigma) ** 2) / max(sigma ** 2, 1e-6)
        weight = min(snr, 1.0)
    else:  # uniform
        weight = 1.0
    loss = base * weight
    return loss, pred_x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dir", default="/home/whadmin/zz/omni_flashtalk_data/distill2/teacher_clean")
    ap.add_argument("--wan-base", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    ap.add_argument("--omni-lora", default="/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--overfit-one", action="store_true",
                    help="lock to item 0 + fixed t_idx + fixed noise for clean determinism")
    ap.add_argument("--fixed-t-idx", type=int, default=1,
                    help="overfit-one mode: which t_idx (0..3) to lock to")
    ap.add_argument("--motion-latent-frames", type=int, default=1)
    ap.add_argument("--loss-weighting", default="uniform",
                    choices=["uniform", "sigma_sq", "inv_sig", "min_snr"],
                    help="loss weighting scheme; sigma_sq up-weights high-noise targets")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--save-dir", default="/home/whadmin/zz/omni_flashtalk_data/distill2/student_ckpts")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"  schedule: timesteps={[round(t,1) for t in DENOISING_STEPS]}  "
          f"sigmas={[round(s,3) for s in SIGMAS]}")

    model = OmniAudioCausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=TEXT_LEN,
        in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=TEXT_DIM,
        out_dim=16, num_heads=12, num_layers=30,
        local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-6)
    model.num_frame_per_block = NUM_FRAME_PER_BLOCK
    model = load_omni_into_causal_adapter(model, args.wan_base, args.omni_lora)
    model = model.to(device=device, dtype=torch.bfloat16).train()
    model.gradient_checkpointing = True

    # Train LoRA + audio + patch_embedding (audio modules included for compatibility
    # though we're not using audio_emb; some norm/bias params are still in the graph)
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n or "audio_" in n or "patch_embedding" in n)
    # Trainable -> fp32 for stable AdamW
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  trainable: {n_trainable/1e6:.1f}M params")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    ds = TeacherDataset(args.teacher_dir)
    assert len(ds) > 0, f"no teacher items in {args.teacher_dir}; run precompute_teacher_latents.py first"

    if args.overfit_one:
        fixed_batch = {k: v.unsqueeze(0) for k, v in ds[0].items()}
        print(f"  OVERFIT-ONE: item 0, fixed t_idx={args.fixed_t_idx} "
              f"(sigma={SIGMAS[args.fixed_t_idx]:.3f}); loss MUST drop")
    else:
        loader = iter(DataLoader(ds, batch_size=1, shuffle=True, num_workers=0))

    losses = []
    t_start = time.time()
    for step in range(args.steps):
        if args.overfit_one:
            batch = fixed_batch
            t_idx = args.fixed_t_idx
            noise_seed = 12345   # fixed noise too for full determinism
        else:
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(DataLoader(ds, batch_size=1, shuffle=True, num_workers=0))
                batch = next(loader)
            t_idx = int(torch.randint(0, len(DENOISING_STEPS), (1,)).item())
            noise_seed = None

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = compute_loss(
                model, batch, t_idx,
                motion_latent_frames=args.motion_latent_frames,
                noise_seed=noise_seed,
                loss_weighting=args.loss_weighting)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append((step, t_idx, loss.item()))

        if step % args.log_every == 0 or step == args.steps - 1:
            recent = [l for _, _, l in losses[-args.log_every:]]
            avg = sum(recent) / len(recent)
            print(f"  step {step:4d}  t_idx={t_idx}  loss={loss.item():.5f}  "
                  f"avg_last{len(recent)}={avg:.5f}  elapsed={time.time()-t_start:.0f}s",
                  flush=True)

        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            # Save only trainable params (LoRA + audio + patch_embedding)
            ckpt = {n: p.detach().cpu() for n, p in model.named_parameters()
                    if p.requires_grad}
            ckpt_path = os.path.join(args.save_dir, f"step_{step+1:05d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"  saved {ckpt_path} ({len(ckpt)} tensors)", flush=True)

    # Final loss summary
    if len(losses) > 20:
        first_20 = sum(l for _, _, l in losses[:20]) / 20
        last_20 = sum(l for _, _, l in losses[-20:]) / 20
        print(f"\n  loss avg first-20: {first_20:.5f}  -> last-20: {last_20:.5f}  "
              f"({100*(last_20-first_20)/first_20:+.1f}%)")
    # Dump loss curve
    with open(os.path.join(args.save_dir, "losses.json"), "w") as f:
        json.dump(losses, f)
    print(f"  saved losses.json")


if __name__ == "__main__":
    main()
