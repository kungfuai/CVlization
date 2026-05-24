"""S5/S6: Stage-1 KD trainer for omni_flashtalk.

Trains the OmniAvatar-1.3B causal student to reproduce the SoulX teacher's
latents via rectified-flow regression.

  x0    = F-frame crop of a SoulX target video latent     [16, F, h, w]
  noise ~ N(0,1)
  t     ~ U(0,1) (shifted)
  xt    = (1-t) x0 + t noise
  v     = noise - x0                                      (velocity target)
  input = cat([xt, ref.tile(F), mask]) -> 33-ch DiT input
  loss  = MSE(student(input, t, zero_text, audio), v)

--overfit-one: lock to item 0 with a FIXED (crop, timestep, noise) so the
loss is fully deterministic — it MUST drop monotonically unless gradients
are dead. The run also logs per-group grad norm + weight movement, to tell a
learning-rate problem from a dead-gradient bug.
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

F_CROP = 18
NUM_FRAME_PER_BLOCK = 3
T_AUDIO = 4 * F_CROP - 3
TEXT_LEN, TEXT_DIM = 512, 4096
TIMESTEP_SHIFT = 5.0
EVAL_TIMESTEPS = [0.1, 0.3, 0.5, 0.7, 0.9]


class Stage1Dataset(Dataset):
    def __init__(self, latents_dir):
        files = sorted(glob.glob(os.path.join(latents_dir, "*.pt")))
        self.files = []
        for f in files:
            b = torch.load(f, map_location="cpu", weights_only=False)
            if b["video_latent"].shape[1] >= F_CROP and "audio_emb" in b:
                self.files.append(f)
        print(f"  Stage1Dataset: {len(self.files)} usable items")

    def __len__(self):
        return len(self.files)

    def _crop(self, b, s):
        vl, ref, aud = b["video_latent"].float(), b["ref_latent"].float(), b["audio_emb"].float()
        x0 = vl[:, s:s + F_CROP]
        a = aud[s * 4:s * 4 + T_AUDIO]
        if a.shape[0] < T_AUDIO:
            a = F.pad(a, (0, 0, 0, T_AUDIO - a.shape[0]))
        return {"x0": x0, "ref": ref[:, 0], "audio": a}

    def __getitem__(self, idx):
        b = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        T_lat = b["video_latent"].shape[1]
        s = torch.randint(0, T_lat - F_CROP + 1, (1,)).item()
        return self._crop(b, s)

    def fixed_item(self, idx):
        b = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        return self._crop(b, 0)


def make_dit_input(xt, ref, mask):
    ref_tiled = ref.unsqueeze(1).expand(-1, xt.shape[1], -1, -1)
    return torch.cat([xt, ref_tiled, mask], dim=0)


def compute_loss(model, x0, ref, audio, t, noise, loss_kind="v", use_audio=True):
    """loss_kind:
       'v'  - MSE(model_out, noise - x0)   (original; my recipe)
       'x0' - MSE(xt - sigma*model_out, x0) (Self-Forcing's ODE-regression recipe)
    """
    B, _, Fc, h, w = x0.shape
    device = x0.device
    dtype = x0.dtype  # follow input precision so mask/ctx match model
    tb = t.view(B, 1, 1, 1, 1)
    xt = (1 - tb) * x0 + tb * noise
    v_target = noise - x0
    mask = torch.zeros(B, 1, Fc, h, w, device=device, dtype=dtype)
    mask[:, :, 1:] = 1.0
    dit_in = [make_dit_input(xt[i], ref[i], mask[i]) for i in range(B)]
    ctx = [torch.zeros(TEXT_LEN, TEXT_DIM, device=device, dtype=dtype) for _ in range(B)]
    t_frames = (t.view(B, 1) * 1000).expand(B, F_CROP)
    seq_len = F_CROP * h * w // 4
    pred = model(dit_in, t_frames, ctx, seq_len, audio_emb=audio if use_audio else None)
    if loss_kind == "x0":
        pred_x0 = xt - tb * pred  # convert flow_pred -> x0 (Self-Forcing convention)
        return F.mse_loss(pred_x0.float(), x0.float())
    return F.mse_loss(pred.float(), v_target.float())


def group_params(model):
    """{group: [params]} for the trainable subset."""
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


def wdelta(params, init_flat):
    cur = torch.cat([p.detach().float().flatten() for p in params])
    return float((cur - init_flat).norm())


def main():
    global F_CROP, T_AUDIO
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/whadmin/zz/omni_flashtalk_data")
    ap.add_argument("--wan-base", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    ap.add_argument("--omni-lora", default="/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--warmup-steps", type=int, default=200,
                    help="linear lr warmup from 0 -> --lr over this many steps")
    ap.add_argument("--overfit-one", action="store_true",
                    help="lock to item 0 + fixed (crop,t,noise) — deterministic loss")
    ap.add_argument("--pure-fp32", action="store_true",
                    help="model stays fp32, no autocast, no grad checkpointing — clean baseline")
    ap.add_argument("--f-crop", type=int, default=F_CROP,
                    help="frames per training crop (default 18; use 6-9 for pure-fp32 to fit 40GB)")
    ap.add_argument("--loss-kind", choices=["v", "x0"], default="v",
                    help="v = velocity MSE (mine); x0 = Self-Forcing's ODE-regression form")
    ap.add_argument("--full-finetune", action="store_true",
                    help="train ALL params (Self-Forcing style); default trains only LoRA+audio+patch")
    ap.add_argument("--no-audio", action="store_true",
                    help="bypass audio path entirely (audio_emb=None); diagnostic")
    args = ap.parse_args()
    F_CROP = args.f_crop
    T_AUDIO = 4 * F_CROP - 3

    torch.manual_seed(0)
    device = "cuda"

    model = OmniAudioCausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=TEXT_LEN,
        in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=TEXT_DIM,
        out_dim=16, num_heads=12, num_layers=30,
        local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-6,
    )
    model.num_frame_per_block = NUM_FRAME_PER_BLOCK
    model = load_omni_into_causal_adapter(model, args.wan_base, args.omni_lora)
    if args.pure_fp32:
        model = model.to(device=device, dtype=torch.float32).train()
        model.gradient_checkpointing = True  # need ckpt to fit; fp32 ckpt is correct
    else:
        model = model.to(device=device, dtype=torch.bfloat16).train()
        model.gradient_checkpointing = True

    for n, p in model.named_parameters():
        if args.full_finetune:
            p.requires_grad_(True)
        else:
            p.requires_grad_("lora_" in n or "audio_" in n or "patch_embedding" in n)
    # Precision fix: when only a small subset is trainable, cast trainable to fp32
    # for stable AdamW + grads while frozen base stays bf16. Skip for full-finetune
    # since fp32 grads + Adam states for 1.3B don't fit on a 40GB card.
    if not args.full_finetune:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    groups = group_params(model)
    trainable = [p for g in groups.values() for p in g]
    init_flat = {k: torch.cat([p.detach().float().flatten() for p in v])
                 for k, v in groups.items() if v}
    print(f"  trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M  "
          f"groups: {{{', '.join(f'{k}:{sum(p.numel() for p in v)/1e6:.0f}M' for k,v in groups.items())}}}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps))
    )
    ds = Stage1Dataset(os.path.join(args.data_dir, "latents"))

    # ---- fixed batch for overfit mode (deterministic loss) ----
    fixed = None
    if args.overfit_one:
        it = ds.fixed_item(0)
        g = torch.Generator(device=device).manual_seed(1234)
        x0 = it["x0"].unsqueeze(0).to(device, torch.bfloat16)
        fixed = dict(
            x0=x0,
            ref=it["ref"].unsqueeze(0).to(device, torch.bfloat16),
            audio=it["audio"].unsqueeze(0).to(device, torch.bfloat16),
            noise=torch.randn(x0.shape, generator=g, device=device, dtype=torch.bfloat16),
            t=torch.full((1,), 0.5, device=device, dtype=torch.bfloat16),
        )
        print("  OVERFIT-ONE: item 0, fixed crop@0, t=0.5, fixed noise "
              "(loss is deterministic — must drop monotonically)")
    else:
        loader = iter(DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, drop_last=True))

    t_start = time.time()
    for step in range(args.steps):
        if args.overfit_one:
            x0, ref, audio = fixed["x0"], fixed["ref"], fixed["audio"]
            noise, t = fixed["noise"], fixed["t"]
        else:
            try:
                batch = next(loader)
            except (StopIteration, NameError):
                loader = iter(DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, drop_last=True))
                batch = next(loader)
            x0 = batch["x0"].to(device, torch.bfloat16)
            ref = batch["ref"].to(device, torch.bfloat16)
            audio = batch["audio"].to(device, torch.bfloat16)
            noise = torch.randn_like(x0)
            t = torch.rand(1, device=device, dtype=torch.bfloat16)
            t = TIMESTEP_SHIFT * t / (1 + (TIMESTEP_SHIFT - 1) * t)

        if args.pure_fp32:
            # cast batch to fp32 to match the fp32 model
            x0 = x0.float(); ref = ref.float(); audio = audio.float()
            noise = noise.float(); t = t.float()
            loss = compute_loss(model, x0, ref, audio, t, noise, args.loss_kind, not args.no_audio)
        else:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compute_loss(model, x0, ref, audio, t, noise, args.loss_kind, not args.no_audio)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = {k: gnorm(v) for k, v in groups.items() if v}
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            wd = {k: wdelta(groups[k], init_flat[k]) for k in init_flat}
            gs = " ".join(f"{k}:g={gn[k]:.2e},dw={wd[k]:.2e}" for k in gn)
            cur_lr = opt.param_groups[0]["lr"]
            print(f"  step {step:4d}  lr={cur_lr:.2e}  loss={loss.item():.4f}  {gs}", flush=True)

    dt = time.time() - t_start
    print(f"\n{args.steps} steps in {dt/60:.1f} min ({dt/args.steps:.2f}s/step)")


if __name__ == "__main__":
    main()
