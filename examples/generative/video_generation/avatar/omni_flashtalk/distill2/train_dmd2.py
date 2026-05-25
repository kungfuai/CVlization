"""DMD2 (Distribution Matching Distillation 2) for talking-avatar.

Three models share OmniAvatar 1.3B base; differ only in LoRA state:
  TEACHER:   OmniAvatar pretrained LoRA, FROZEN. Provides real score.
  GENERATOR: starts from OmniAvatar pretrained, trainable. The student we distill.
  FAKE_SCORE: starts from OmniAvatar pretrained, trainable. Estimates current
              generator's distribution score.

Per iteration:
  z ~ N(0,I)                                        # pure noise
  x_pred = GENERATOR(z, t=1.0)                       # 1-step prediction (with grad)
  sigma_t ~ U(0,1)
  noise2 = randn_like(x_pred)
  xt = (1-sigma_t)*x_pred + sigma_t*noise2          # forward-noise to sigma_t

  with no_grad:
    x_real = TEACHER(xt, sigma_t)                   # teacher's predicted x0
    x_fake = FAKE_SCORE(xt, sigma_t)                # critic's predicted x0

  # Generator loss: push x_pred toward where teacher says yes and fake_score says no.
  # Straight-through gradient via stop-gradient on the score difference.
  weight = 1 / max(sigma_t, 0.1)
  grad_target = weight * (x_fake - x_real).detach()
  L_gen = (grad_target * x_pred).mean()             # backprop through generator only

  # Fake_score loss: critic should denoise the generator's outputs.
  with no_grad: x_pred_detached = x_pred.detach()
  xt2 = (1-sigma_t) * x_pred_detached + sigma_t * noise2  # same xt, but with detached x
  x_fake2 = FAKE_SCORE(xt2, sigma_t)                # with grad
  L_fake = MSE(x_fake2, x_pred_detached)            # standard denoising loss

  opt_gen.step(L_gen)
  opt_fake.step(L_fake)

This is the textbook DMD2 algorithm (Yin et al. 2024). The key property:
unlike MSE distillation, the (fake - real) score signal is well-defined and
informative even at high sigma_t where MSE has weak gradient.
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

# Generator runs at sigma=1.0 (one-step from pure noise). For fake_score and
# teacher we sample sigma_t uniformly over the 4-step schedule (and these
# values are what the model was trained on so they're in-distribution).
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
    """Just provides text + ref + (optional) one clean target latent as a real-image
    sample. We mostly need text+ref for conditioning; the clean latent is used
    as a real-sample anchor for fake_score warmup."""

    def __init__(self, teacher_dir):
        self.files = sorted(glob.glob(os.path.join(teacher_dir, "*.pt")))
        print(f"  TeacherDataset: {len(self.files)} items")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        return {
            "clean_latent": b["clean_latent"].float(),
            "ref_latent":   b["ref_latent"].float(),
            "text_context": b["text_context"].float(),
        }


def make_model(wan_base, omni_lora, device):
    """Construct one OmniAvatar model with pretrained LoRA loaded."""
    m = OmniAudioCausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=TEXT_LEN,
        in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=TEXT_DIM,
        out_dim=16, num_heads=12, num_layers=30,
        local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-6)
    m.num_frame_per_block = NUM_FRAME_PER_BLOCK
    m = load_omni_into_causal_adapter(m, wan_base, omni_lora)
    m = m.to(device=device, dtype=torch.bfloat16)
    return m


def model_forward(model, xt, t_val, ref, text, motion_latent_frames):
    """One forward pass of OmniAvatar (no audio). xt is [B,16,F,H,W]; returns
    flow_pred [B,16,F,H,W]."""
    B, _, Fc, H, W = xt.shape
    dtype = xt.dtype
    device = xt.device
    mask = torch.zeros(B, 1, Fc, H, W, device=device, dtype=dtype)
    mask[:, :, motion_latent_frames:] = 1.0
    ref_tiled = ref.unsqueeze(2).expand(-1, -1, Fc, -1, -1)
    dit_in = [torch.cat([xt[i], ref_tiled[i], mask[i]], dim=0) for i in range(B)]
    ctx = [text[i] for i in range(B)]
    t_frames = torch.full((B, Fc), t_val, device=device, dtype=dtype)
    seq_len = Fc * H * W // 4
    return model(dit_in, t_frames, ctx, seq_len, audio_emb=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dir", default="/home/whadmin/zz/omni_flashtalk_data/distill2/teacher_clean")
    ap.add_argument("--wan-base", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    ap.add_argument("--omni-lora", default="/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr-gen", type=float, default=1e-5)
    ap.add_argument("--lr-fake", type=float, default=1e-5)
    ap.add_argument("--fake-warmup-steps", type=int, default=50,
                    help="train only fake_score for this many steps first; helps stability")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--motion-latent-frames", type=int, default=1)
    ap.add_argument("--save-dir", default="/home/whadmin/zz/omni_flashtalk_data/distill2/dmd2")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"  schedule sigmas: {[round(s,3) for s in SIGMAS]}")

    # ---- 3 model copies ----------------------------------------------------
    print("loading TEACHER...")
    teacher = make_model(args.wan_base, args.omni_lora, device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("loading GENERATOR...")
    generator = make_model(args.wan_base, args.omni_lora, device).train()
    generator.gradient_checkpointing = True
    # Only LoRA + audio + patch_embedding trainable
    for n, p in generator.named_parameters():
        p.requires_grad_("lora_" in n or "audio_" in n or "patch_embedding" in n)
    for p in generator.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    gen_trainable = [p for p in generator.parameters() if p.requires_grad]
    print(f"  generator trainable: {sum(p.numel() for p in gen_trainable)/1e6:.1f}M")

    print("loading FAKE_SCORE...")
    fake_score = make_model(args.wan_base, args.omni_lora, device).train()
    fake_score.gradient_checkpointing = True
    for n, p in fake_score.named_parameters():
        p.requires_grad_("lora_" in n or "audio_" in n or "patch_embedding" in n)
    for p in fake_score.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    fake_trainable = [p for p in fake_score.parameters() if p.requires_grad]
    print(f"  fake_score trainable: {sum(p.numel() for p in fake_trainable)/1e6:.1f}M")

    print(f"  GPU memory after model load: "
          f"{torch.cuda.memory_allocated()/1e9:.1f}GB / "
          f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB peak")

    opt_gen = torch.optim.AdamW(gen_trainable, lr=args.lr_gen, betas=(0.9, 0.999))
    opt_fake = torch.optim.AdamW(fake_trainable, lr=args.lr_fake, betas=(0.9, 0.999))

    ds = TeacherDataset(args.teacher_dir)
    assert len(ds) > 0, f"no teacher items in {args.teacher_dir}"
    loader = iter(DataLoader(ds, batch_size=1, shuffle=True, num_workers=0))

    history = []
    t_start = time.time()
    for step in range(args.steps):
        try:
            batch = next(loader)
        except StopIteration:
            loader = iter(DataLoader(ds, batch_size=1, shuffle=True, num_workers=0))
            batch = next(loader)

        clean = batch["clean_latent"].to(device, torch.bfloat16)         # [1, 16, F, H, W]
        ref   = batch["ref_latent"][:, :, 0].to(device, torch.bfloat16)  # [1, 16, H, W]
        text  = batch["text_context"].to(device, torch.bfloat16)         # [1, text_len, text_dim]
        B, _, Fc, H, W = clean.shape
        motion = clean[:, :, :args.motion_latent_frames]

        # === Generator forward: 1-step from pure noise ====================
        z = torch.randn_like(clean)
        z[:, :, :args.motion_latent_frames] = motion
        # Generator predicts velocity from pure noise at sigma=1.0
        sigma_init = 1.0
        t_init = DENOISING_STEPS[0]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen_velocity = model_forward(generator, z, t_init, ref, text,
                                          args.motion_latent_frames)
        x_pred = z - sigma_init * gen_velocity                            # [1, 16, F, H, W]
        x_pred = torch.cat([motion, x_pred[:, :, args.motion_latent_frames:]], dim=2)

        # === Sample sigma_t, add noise to x_pred ==========================
        t_idx = int(torch.randint(0, len(SIGMAS), (1,)).item())
        sigma_t = SIGMAS[t_idx]
        t_val = DENOISING_STEPS[t_idx]
        noise2 = torch.randn_like(x_pred)
        xt = (1 - sigma_t) * x_pred + sigma_t * noise2
        xt = torch.cat([motion, xt[:, :, args.motion_latent_frames:]], dim=2)

        # === Teacher + fake_score scores (no grad through them for L_gen) =
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_real = model_forward(teacher, xt, t_val, ref, text,
                                    args.motion_latent_frames)
            v_fake = model_forward(fake_score, xt, t_val, ref, text,
                                    args.motion_latent_frames)
        x_real = xt - sigma_t * v_real
        x_fake = xt - sigma_t * v_fake

        # === Generator loss (skip during fake_score warmup) ===============
        if step >= args.fake_warmup_steps:
            # straight-through: grad of L_gen w.r.t. x_pred is the (fake-real) target.
            # NO /N here -- gradient per element = grad_target directly, so weight
            # updates are O(grad_target * lr) per param. With grad_target ~ 0.1 and
            # lr=1e-5, per-step param change is ~1e-6, summed across N elements via
            # chain rule gives meaningful magnitudes.
            weight = 1.0 / max(sigma_t, 0.1)
            grad_target = weight * (x_fake - x_real).detach()
            L_gen = (x_pred * grad_target).sum()  # NO mean — keep gradient magnitude
            opt_gen.zero_grad(set_to_none=True)
            L_gen.backward()
            torch.nn.utils.clip_grad_norm_(gen_trainable, 1.0)
            opt_gen.step()
            gen_loss_val = float(L_gen.detach())
        else:
            gen_loss_val = float("nan")

        # === Fake_score loss: denoise generator's outputs ================
        # Re-use the noise2 + sigma_t but on detached x_pred
        x_pred_det = x_pred.detach()
        xt_det = (1 - sigma_t) * x_pred_det + sigma_t * noise2
        xt_det = torch.cat([motion, xt_det[:, :, args.motion_latent_frames:]], dim=2)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_fake_train = model_forward(fake_score, xt_det, t_val, ref, text,
                                          args.motion_latent_frames)
        x_fake_train = xt_det - sigma_t * v_fake_train
        # Loss only on non-motion frames
        pred_eval = x_fake_train[:, :, args.motion_latent_frames:]
        target_eval = x_pred_det[:, :, args.motion_latent_frames:]
        L_fake = F.mse_loss(pred_eval.float(), target_eval.float())
        opt_fake.zero_grad(set_to_none=True)
        L_fake.backward()
        torch.nn.utils.clip_grad_norm_(fake_trainable, 1.0)
        opt_fake.step()
        fake_loss_val = float(L_fake.detach())

        history.append({
            "step": step, "t_idx": t_idx, "sigma": sigma_t,
            "L_gen": gen_loss_val, "L_fake": fake_loss_val,
            "x_pred_std": float(x_pred.detach().float().std()),
            "x_real_std": float(x_real.detach().float().std()),
            "x_fake_std": float(x_fake.detach().float().std()),
        })

        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.time() - t_start
            print(f"  step {step:4d}  t_idx={t_idx}  L_gen={gen_loss_val:+.5f}  "
                  f"L_fake={fake_loss_val:.5f}  x_pred.std={history[-1]['x_pred_std']:.3f}  "
                  f"elapsed={elapsed:.0f}s",
                  flush=True)

        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            ckpt = {n: p.detach().cpu() for n, p in generator.named_parameters()
                    if p.requires_grad}
            ckpt_path = os.path.join(args.save_dir, f"generator_step_{step+1:05d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f)
    print(f"\n  saved history.json")


if __name__ == "__main__":
    main()
