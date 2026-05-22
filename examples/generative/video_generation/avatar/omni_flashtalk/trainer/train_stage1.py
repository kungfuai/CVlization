"""S5/S6: Stage-1 KD trainer for omni_flashtalk.

Trains the OmniAvatar-1.3B causal student to reproduce the SoulX teacher's
latents via rectified-flow regression.

Per step:
  x0     = a F-frame crop of a SoulX target video latent      [16, F, h, w]
  noise  ~ N(0,1)
  t      ~ U(0,1) per item, with timestep shift
  xt     = (1 - t) * x0 + t * noise          (rectified flow interpolant)
  v      = noise - x0                        (the velocity target)
  input  = cat([xt, ref_latent.tile(F), mask], ch) -> 33-ch DiT input
  pred   = student(input, t, zero_text_context, audio_emb)
  loss   = MSE(pred, v)

The raw per-step training loss is dominated by timestep variance (loss at
t~=1 >> loss at t~=0), so it is NOT a learning signal. The eval metric below
is the real signal: a FIXED held-out set of (item, crop, noise, timestep)
tuples, so eval loss is comparable across steps.

Single-GPU; the 1.3B student fits comfortably (~9 GB peak).
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
EVAL_TIMESTEPS = [0.1, 0.3, 0.5, 0.7, 0.9]   # fixed grid for the eval metric


class Stage1Dataset(Dataset):
    def __init__(self, latents_dir):
        files = sorted(glob.glob(os.path.join(latents_dir, "*.pt")))
        self.files = []
        for f in files:
            b = torch.load(f, map_location="cpu", weights_only=False)
            if b["video_latent"].shape[1] >= F_CROP and "audio_emb" in b:
                self.files.append(f)
        print(f"  Stage1Dataset: {len(self.files)} usable items (T_lat >= {F_CROP})")

    def __len__(self):
        return len(self.files)

    def _crop(self, b, s):
        vl = b["video_latent"].float()
        ref = b["ref_latent"].float()
        aud = b["audio_emb"].float()
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
        """Deterministic crop from frame 0 — used to build the eval set."""
        b = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        return self._crop(b, 0)


def make_dit_input(xt, ref, mask):
    F_ = xt.shape[1]
    ref_tiled = ref.unsqueeze(1).expand(-1, F_, -1, -1)
    return torch.cat([xt, ref_tiled, mask], dim=0)


def compute_loss(model, x0, ref, audio, t, noise):
    """Rectified-flow regression loss for a batch at given timesteps t."""
    B, _, Fc, h, w = x0.shape
    device = x0.device
    tb = t.view(B, 1, 1, 1, 1)
    xt = (1 - tb) * x0 + tb * noise
    v_target = noise - x0
    mask = torch.zeros(B, 1, Fc, h, w, device=device, dtype=torch.bfloat16)
    mask[:, :, 1:] = 1.0
    dit_in = [make_dit_input(xt[i], ref[i], mask[i]) for i in range(B)]
    ctx = [torch.zeros(TEXT_LEN, TEXT_DIM, device=device, dtype=torch.bfloat16)
           for _ in range(B)]
    t_frames = (t.view(B, 1) * 1000).expand(B, F_CROP)
    seq_len = F_CROP * h * w // (1 * 2 * 2)
    pred = model(dit_in, t_frames, ctx, seq_len, audio_emb=audio)
    return F.mse_loss(pred.float(), v_target.float())


@torch.no_grad()
def evaluate(model, eval_set):
    """Mean loss over the FIXED eval set — comparable across steps."""
    model.eval()
    total, n = 0.0, 0
    for x0, ref, audio, noise in eval_set:
        for tv in EVAL_TIMESTEPS:
            t = torch.full((1,), tv, device=x0.device, dtype=torch.bfloat16)
            total += compute_loss(model, x0, ref, audio, t, noise).item()
            n += 1
    model.train()
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/whadmin/zz/omni_flashtalk_data")
    ap.add_argument("--wan-base", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    ap.add_argument("--omni-lora", default="/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-items", type=int, default=8)
    args = ap.parse_args()

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
    model = model.to(device=device, dtype=torch.bfloat16).train()
    model.gradient_checkpointing = True

    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n or "audio_" in n or "patch_embedding" in n)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M params")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    ds = Stage1Dataset(os.path.join(args.data_dir, "latents"))

    # ---- fixed eval set: items [0:eval_items], crop@0, frozen noise ----
    g = torch.Generator(device=device).manual_seed(1234)
    eval_set = []
    for i in range(min(args.eval_items, len(ds))):
        it = ds.fixed_item(i)
        x0 = it["x0"].unsqueeze(0).to(device, torch.bfloat16)
        ref = it["ref"].unsqueeze(0).to(device, torch.bfloat16)
        audio = it["audio"].unsqueeze(0).to(device, torch.bfloat16)
        noise = torch.randn(x0.shape, generator=g, device=device, dtype=torch.bfloat16)
        eval_set.append((x0, ref, audio, noise))
    print(f"  eval set: {len(eval_set)} items x {len(EVAL_TIMESTEPS)} timesteps")

    loader = iter(DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, drop_last=True))

    def next_batch():
        nonlocal loader
        try:
            return next(loader)
        except StopIteration:
            loader = iter(DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, drop_last=True))
            return next(loader)

    eval0 = evaluate(model, eval_set)
    print(f"  eval@0 (pre-training): {eval0:.4f}", flush=True)

    t_start = time.time()
    for step in range(args.steps):
        batch = next_batch()
        x0 = batch["x0"].to(device, torch.bfloat16)
        ref = batch["ref"].to(device, torch.bfloat16)
        audio = batch["audio"].to(device, torch.bfloat16)
        B = x0.shape[0]
        noise = torch.randn_like(x0)
        t = torch.rand(B, device=device, dtype=torch.bfloat16)
        t = TIMESTEP_SHIFT * t / (1 + (TIMESTEP_SHIFT - 1) * t)

        loss = compute_loss(model, x0, ref, audio, t, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % args.log_every == 0:
            print(f"  step {step:4d}  train_loss={loss.item():.4f}", flush=True)
        if (step + 1) % args.eval_every == 0:
            ev = evaluate(model, eval_set)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  step {step+1:4d}  EVAL={ev:.4f}  ({peak:.1f}GB)", flush=True)

    dt = time.time() - t_start
    evN = evaluate(model, eval_set)
    print(f"\ntrained {args.steps} steps in {dt/60:.1f} min ({dt/args.steps:.2f}s/step)")
    print(f"  eval: pre={eval0:.4f}  final={evN:.4f}  "
          f"delta={(evN-eval0):.4f} ({100*(evN-eval0)/eval0:+.1f}%)")

    out = os.path.join(args.data_dir, "stage1_ckpt.pt")
    torch.save({n: p for n, p in model.named_parameters() if p.requires_grad}, out)
    print(f"  saved trainable params -> {out}")


if __name__ == "__main__":
    main()
