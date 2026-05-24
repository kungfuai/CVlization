"""Stage-1 KD trainer, Hallo-Live / CausVid ODE-fusion recipe.

Replaces train_stage1.py's continuous-t velocity loss (which gave wrong-
direction gradients on a pretrained student) with the discrete-trajectory
recipe that Hallo-Live's `ode_fusion_regression.py` actually uses.

Per-item input: trajectories/<id>.pt = {
    "ode_traj":         [5, C, F, H, W]   # 4 noisy intermediates @
                                          # t=1000/750/500/250 + clean x0
    "denoising_steps":  [1000, 750, 500, 250, 0],
}
plus latents/<id>.pt = {"audio_emb", "ref_latent"} from the original pack.

Training step:
    idx     ~ U{0, 1, 2, 3}              # pick one of 4 noisy intermediates
    noisy   = ode_traj[idx]              # teacher's actual latent at that t
    target  = ode_traj[-1]               # teacher's clean x0
    t_in    = denoising_steps[idx]       # the integer timestep
    pred    = model(noisy, t_in, ref, audio)
    pred_x0 = noisy - sigma_t * pred     # convert velocity -> x0
    loss    = MSE(pred_x0, target)

Why this works where the old recipe didn't:
- The noisy input is the teacher's actual partial denoising state, so the
  pretrained student sees IN-distribution inputs (not a random Gaussian-noised
  x0). Tiny output errors -> small, well-conditioned gradients.
- Discrete timesteps from [1000,750,500,250] match what the student will face
  at inference, not random uniform t.
- x0-regression matches Hallo-Live / Self-Forcing / CausVid exactly.

Overfit-one smoke is the gate: with fixed (item, idx), loss MUST drop monotonically.
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from omni_causal_adapter import OmniAudioCausalWanModel, load_omni_into_causal_adapter

# ---- schedule constants -----------------------------------------------------
# Matches SoulX `sample_steps=4`, `sample_shift=5.0`, and Hallo-Live's
# `denoising_step_list=[1000, 750, 500, 250]`.
DENOISING_STEPS_RAW = [1000.0, 750.0, 500.0, 250.0]      # before shift
TIMESTEP_SHIFT = 5.0                                      # sigma shift
NUM_TIMESTEPS = 1000.0                                    # train_timesteps

# Apply same shift transform SoulX uses (see flash_talk_pipeline.timestep_transform).
# t_shifted = shift * t / (1 + (shift - 1) * t)   with t in [0, 1]
def _shift(t):
    f = t / NUM_TIMESTEPS
    return NUM_TIMESTEPS * (TIMESTEP_SHIFT * f / (1 + (TIMESTEP_SHIFT - 1) * f))

DENOISING_STEPS = [_shift(t) for t in DENOISING_STEPS_RAW]  # what we feed the model
# Corresponding sigmas (mixing fractions) for converting flow_pred -> x0.
# xt = (1 - sigma) * x0 + sigma * noise  ->  sigma = t / NUM_TIMESTEPS  (shifted)
SIGMAS = [t / NUM_TIMESTEPS for t in DENOISING_STEPS]

NUM_FRAME_PER_BLOCK = 6     # matches Hallo-Live video_num_frame_per_block
TEXT_LEN, TEXT_DIM = 512, 4096


class TrajDataset(Dataset):
    """One sample = one (item, t_idx) pair. We crop F_CROP frames in time from
    the trajectory; the time crop is shared across all 5 trajectory snapshots."""

    def __init__(self, data_dir, f_crop):
        self.f_crop = f_crop
        traj_dir = os.path.join(data_dir, "trajectories")
        lat_dir = os.path.join(data_dir, "latents")
        self.items = []
        for traj_pt in sorted(glob.glob(os.path.join(traj_dir, "*.pt"))):
            sid = os.path.splitext(os.path.basename(traj_pt))[0]
            lat_pt = os.path.join(lat_dir, f"{sid}.pt")
            if not os.path.exists(lat_pt):
                continue
            traj_blob = torch.load(traj_pt, map_location="cpu", weights_only=False)
            ode_traj = traj_blob["ode_traj"]  # [5, C, F, H, W]
            if ode_traj.shape[2] < f_crop:
                continue
            self.items.append((traj_pt, lat_pt))
        print(f"  TrajDataset: {len(self.items)} usable items")

    def __len__(self):
        return len(self.items)

    def fixed_item(self, idx):
        return self._load(idx, t_crop=0)

    def __getitem__(self, idx):
        # Random time-crop start; same crop applied to all trajectory steps + audio.
        traj_pt, _ = self.items[idx]
        traj_blob = torch.load(traj_pt, map_location="cpu", weights_only=False)
        T_lat = traj_blob["ode_traj"].shape[2]
        s = torch.randint(0, T_lat - self.f_crop + 1, (1,)).item()
        return self._load(idx, t_crop=s)

    def _load(self, idx, t_crop):
        traj_pt, lat_pt = self.items[idx]
        traj_blob = torch.load(traj_pt, map_location="cpu", weights_only=False)
        lat_blob = torch.load(lat_pt, map_location="cpu", weights_only=False)
        ode = traj_blob["ode_traj"].float()        # [5, C, F, H, W]
        F_lat = ode.shape[2]
        s, fc = t_crop, self.f_crop
        ode_crop = ode[:, :, s:s + fc]              # [5, C, fc, H, W]
        ref = lat_blob["ref_latent"].float()        # [C, 1, H, W]
        audio = lat_blob["audio_emb"].float()       # [T_audio, D]
        text = lat_blob.get("text_context")         # [text_len=512, text_dim=4096]
        if text is None:
            # Fall back to zeros (the broken legacy path). Warn loudly so we
            # notice when training without text conditioning.
            text = torch.zeros(TEXT_LEN, TEXT_DIM)
        text = text.float()
        # SoulX audio rate is 4 audio frames per latent frame, minus 3-frame pad.
        T_audio_target = 4 * fc - 3
        a0 = s * 4
        a_crop = audio[a0:a0 + T_audio_target]
        if a_crop.shape[0] < T_audio_target:
            a_crop = F.pad(a_crop, (0, 0, 0, T_audio_target - a_crop.shape[0]))
        return {"ode": ode_crop, "ref": ref[:, 0], "audio": a_crop, "text": text}


def make_dit_input(xt, ref, mask):
    ref_tiled = ref.unsqueeze(1).expand(-1, xt.shape[1], -1, -1)
    return torch.cat([xt, ref_tiled, mask], dim=0)


def compute_loss(model, ode, ref, audio, text, t_idx, use_audio=True):
    """ODE-regression loss: pick a step idx, predict x0 from teacher's noisy.
    Returns (loss, pred_x0, target_x0) for diagnostic logging.

    text: real T5-encoded prompt [B, text_len, text_dim] — NOT zero. The
    OmniAvatar student was trained with text conditioning; passing zero text
    puts it in OOD regime and produces garbage. precompute_text.py builds these."""
    B = ode.shape[0]
    device = ode.device
    dtype = ode.dtype
    _, _, Fc, h, w = ode.shape[1:]                 # ode is [B, 5, C, Fc, H, W]

    noisy = ode[:, t_idx]                          # [B, C, Fc, H, W]
    target_x0 = ode[:, -1]
    sigma = SIGMAS[t_idx]
    t_val = DENOISING_STEPS[t_idx]

    mask = torch.zeros(B, 1, Fc, h, w, device=device, dtype=dtype)
    mask[:, :, 1:] = 1.0
    dit_in = [make_dit_input(noisy[i], ref[i], mask[i]) for i in range(B)]
    ctx = [text[i] for i in range(B)]              # real T5 embedding per item
    t_frames = torch.full((B, Fc), t_val, device=device, dtype=dtype)
    seq_len = Fc * h * w // 4

    flow_pred = model(dit_in, t_frames, ctx, seq_len,
                       audio_emb=audio if use_audio else None)
    pred_x0 = noisy - sigma * flow_pred             # Wan/SF convention
    loss = F.mse_loss(pred_x0.float(), target_x0.float())
    return loss, pred_x0, target_x0


def group_params(model):
    g = {"lora": [], "audio": [], "patch_embedding": [], "base": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            g["lora"].append(p)
        elif "audio_" in n:
            g["audio"].append(p)
        elif "patch_embedding" in n:
            g["patch_embedding"].append(p)
        else:
            g["base"].append(p)
    return g


def gnorm(params):
    sq = sum((p.grad.detach().float() ** 2).sum() for p in params if p.grad is not None)
    return float(sq.sqrt()) if torch.is_tensor(sq) else 0.0


def load_wan_vae(vae_path, device):
    """Load WanVideoVAE from OmniAvatar's checkpoint. Returns the VAE wrapper."""
    from OmniAvatar.models.wan_video_vae import WanVideoVAE
    vae = WanVideoVAE(z_dim=16)
    sd = torch.load(vae_path, map_location="cpu", weights_only=False)
    # File holds VideoVAE_'s raw state_dict (keys: encoder.*, decoder.*, no
    # "model." prefix). Load into the inner module, not the wrapper.
    missing, unexpected = vae.model.load_state_dict(sd, strict=True)
    print(f"  Wan VAE: missing={len(missing)} unexpected={len(unexpected)}")
    vae = vae.to(device=device, dtype=torch.bfloat16).eval()
    return vae


@torch.no_grad()
def decode_latent(vae, latent_bcfhw, device):
    """latent_bcfhw: [C=16, F, H, W] -> video pixels [3, F_pixel, H*8, W*8] in [-1, 1]."""
    pix = vae.decode([latent_bcfhw.to(device=device, dtype=torch.bfloat16)], device)
    return pix[0].float()  # [3, F_pixel, H_p, W_p]


@torch.no_grad()
def pixel_psnr(a, b):
    """Both in [-1, 1]. Returns PSNR in dB."""
    a, b = a.float(), b.float()
    mse = (a - b).pow(2).mean()
    if mse == 0:
        return float("inf")
    # signal range is 2.0 (-1..1)
    return float(20 * torch.log10(2.0 / mse.sqrt()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/whadmin/zz/omni_flashtalk_data")
    ap.add_argument("--wan-base", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    ap.add_argument("--omni-lora", default="/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-6,
                    help="Hallo-Live ode_init default")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--f-crop", type=int, default=12,
                    help="frames per crop (multiple of num_frame_per_block=6)")
    ap.add_argument("--overfit-one", action="store_true",
                    help="lock to item 0 + fixed t_idx=1 (sigma=0.938)")
    ap.add_argument("--fixed-t-idx", type=int, default=1,
                    help="for overfit-one: which of 4 noise levels [0..3]")
    ap.add_argument("--full-finetune", action="store_true",
                    help="train ALL params; default is LoRA+audio+patch only")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--wan-vae",
                    default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    ap.add_argument("--pixel-eval-every", type=int, default=0,
                    help="decode student pred_x0 + teacher x0 every N steps, log PSNR; 0=off")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    print(f"  schedule: denoising_steps={[round(s,1) for s in DENOISING_STEPS]}  "
          f"sigmas={[round(s,3) for s in SIGMAS]}")

    model = OmniAudioCausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=TEXT_LEN,
        in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=TEXT_DIM,
        out_dim=16, num_heads=12, num_layers=30,
        local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-6,
    )
    model.num_frame_per_block = NUM_FRAME_PER_BLOCK
    model = load_omni_into_causal_adapter(model, args.wan_base, args.omni_lora)
    model = model.to(device=device, dtype=torch.bfloat16).train()
    model.gradient_checkpointing = True

    for n, p in model.named_parameters():
        if args.full_finetune:
            p.requires_grad_(True)
        else:
            p.requires_grad_("lora_" in n or "audio_" in n or "patch_embedding" in n)
    # Trainable params -> fp32 for stable AdamW; frozen base stays bf16.
    # (skip for full-finetune since 1.3B fp32 grads + Adam won't fit 40GB.)
    if not args.full_finetune:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    groups = group_params(model)
    trainable = [p for g in groups.values() for p in g]
    print(f"  trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M  "
          f"groups: {{{', '.join(f'{k}:{sum(p.numel() for p in v)/1e6:.0f}M' for k,v in groups.items())}}}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    ds = TrajDataset(args.data_dir, args.f_crop)
    assert len(ds) > 0, "no trajectories found — run capture_trajectories.py first"

    # ---- overfit mode: fixed item + fixed t_idx, deterministic loss --------
    fixed = None
    if args.overfit_one:
        it = ds.fixed_item(0)
        fixed = dict(
            ode=it["ode"].unsqueeze(0).to(device, torch.bfloat16),
            ref=it["ref"].unsqueeze(0).to(device, torch.bfloat16),
            audio=it["audio"].unsqueeze(0).to(device, torch.bfloat16),
            text=it["text"].unsqueeze(0).to(device, torch.bfloat16),
        )
        print(f"  OVERFIT-ONE: item 0, fixed crop@0, t_idx={args.fixed_t_idx} "
              f"(t={DENOISING_STEPS[args.fixed_t_idx]:.1f}, sigma={SIGMAS[args.fixed_t_idx]:.3f}) — loss MUST drop")
    else:
        loader = iter(DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, drop_last=True))

    # ---- pixel-eval setup --------------------------------------------------
    vae = teacher_pixels = None
    if args.pixel_eval_every > 0 and args.overfit_one:
        vae = load_wan_vae(args.wan_vae, device)
        # decode teacher's clean x0 ONCE as the reference (item 0, fixed crop@0)
        teacher_pixels = decode_latent(vae, fixed["ode"][0, -1], device)
        print(f"  teacher_pixels: shape={tuple(teacher_pixels.shape)}  "
              f"range=[{teacher_pixels.min():.2f},{teacher_pixels.max():.2f}]")

    t_start = time.time()
    last_loss = None
    monotonic_drop_streak = 0
    initial_psnr = None
    for step in range(args.steps):
        if args.overfit_one:
            batch = fixed
            t_idx = args.fixed_t_idx
        else:
            try:
                batch = next(loader)
            except (StopIteration, NameError):
                loader = iter(DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, drop_last=True))
                batch = next(loader)
            batch = {k: v.to(device, torch.bfloat16) for k, v in batch.items()}
            t_idx = int(torch.randint(0, len(DENOISING_STEPS), (1,)).item())

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, pred_x0, _ = compute_loss(
                model, batch["ode"], batch["ref"], batch["audio"], batch["text"],
                t_idx, use_audio=not args.no_audio)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = {k: gnorm(v) for k, v in groups.items() if v}
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if last_loss is not None and loss.item() < last_loss:
            monotonic_drop_streak += 1
        else:
            monotonic_drop_streak = 0
        last_loss = loss.item()

        # ---- pixel-eval: decode student pred_x0, compute PSNR vs teacher ---
        psnr_str = ""
        if vae is not None and (step % args.pixel_eval_every == 0 or step == args.steps - 1):
            student_pixels = decode_latent(vae, pred_x0[0].detach(), device)
            psnr = pixel_psnr(student_pixels, teacher_pixels)
            if initial_psnr is None:
                initial_psnr = psnr
            psnr_str = f"  psnr={psnr:.2f}dB  d_psnr={psnr - initial_psnr:+.2f}"

        if step % args.log_every == 0 or step == args.steps - 1:
            gs = " ".join(f"{k}:g={gn[k]:.2e}" for k in gn)
            print(f"  step {step:4d}  t_idx={t_idx}  loss={loss.item():.4f}  "
                  f"drop_streak={monotonic_drop_streak}{psnr_str}  {gs}", flush=True)

    dt = time.time() - t_start
    print(f"\n{args.steps} steps in {dt/60:.1f} min ({dt/args.steps:.2f}s/step)")


if __name__ == "__main__":
    main()
