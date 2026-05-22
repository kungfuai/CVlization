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

Flow-matching noise is applied only to the 16-ch video latent; the ref-latent
and mask channels are clean conditioning. Text context is zero for the smoke
(see STAGE1_PLAN.md S4).

Single-GPU; the 1.3B student fits comfortably (cf. E5 = 5.4 GB peak).
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

# ---- training shape constants ----
F_CROP = 18                 # latent frames per training crop (18 % 3 == 0)
NUM_FRAME_PER_BLOCK = 3
T_AUDIO = 4 * F_CROP - 3    # 69 — audio frames the student's _prepare_audio expects
TEXT_LEN, TEXT_DIM = 512, 4096
TIMESTEP_SHIFT = 5.0


class Stage1Dataset(Dataset):
    """Reads latents/<id>.pt = {video_latent, ref_latent, audio_emb}."""

    def __init__(self, latents_dir):
        self.files = sorted(glob.glob(os.path.join(latents_dir, "*.pt")))
        # keep only items long enough to crop
        keep = []
        for f in self.files:
            b = torch.load(f, map_location="cpu", weights_only=False)
            if b["video_latent"].shape[1] >= F_CROP and "audio_emb" in b:
                keep.append(f)
        self.files = keep
        print(f"  Stage1Dataset: {len(self.files)} usable items (T_lat >= {F_CROP})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        vl = b["video_latent"].float()        # [16, T_lat, h, w]
        ref = b["ref_latent"].float()         # [16, 1, h, w]
        aud = b["audio_emb"].float()          # [T_audio_full, 10752]
        T_lat = vl.shape[1]
        s = torch.randint(0, T_lat - F_CROP + 1, (1,)).item()
        x0 = vl[:, s:s + F_CROP]              # [16, F, h, w]
        # audio crop aligned to the latent window (audio frame ~= 4 x latent frame)
        a0 = s * 4
        a = aud[a0:a0 + T_AUDIO]
        if a.shape[0] < T_AUDIO:              # pad tail if short
            a = F.pad(a, (0, 0, 0, T_AUDIO - a.shape[0]))
        return {"x0": x0, "ref": ref[:, 0], "audio": a}


def make_dit_input(xt, ref, mask):
    """cat([noisy video latent(16), ref tiled(16), mask(1)]) -> [33, F, h, w]."""
    F_ = xt.shape[1]
    ref_tiled = ref.unsqueeze(1).expand(-1, F_, -1, -1)   # [16, F, h, w]
    return torch.cat([xt, ref_tiled, mask], dim=0)        # [33, F, h, w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/whadmin/zz/omni_flashtalk_data")
    ap.add_argument("--wan-base", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    ap.add_argument("--omni-lora", default="/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda"

    # ---- model ----
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

    # trainable subset: LoRA + audio path + patch_embedding (cf. E5)
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n or "audio_" in n or "patch_embedding" in n)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"  trainable: {n_tr/1e6:.1f}M params")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    ds = Stage1Dataset(os.path.join(args.data_dir, "latents"))
    loader = iter(DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=2, drop_last=True))

    def next_batch():
        nonlocal loader
        try:
            return next(loader)
        except StopIteration:
            loader = iter(DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                     num_workers=2, drop_last=True))
            return next(loader)

    h, w = None, None
    losses = []
    t_start = time.time()
    for step in range(args.steps):
        batch = next_batch()
        x0 = batch["x0"].to(device, torch.bfloat16)        # [B,16,F,h,w]
        ref = batch["ref"].to(device, torch.bfloat16)      # [B,16,h,w]
        audio = batch["audio"].to(device, torch.bfloat16)  # [B,T_audio,10752]
        B = x0.shape[0]
        h, w = x0.shape[-2], x0.shape[-1]

        noise = torch.randn_like(x0)
        # one timestep per item, shifted; broadcast across frames
        t = torch.rand(B, device=device, dtype=torch.bfloat16)
        t = TIMESTEP_SHIFT * t / (1 + (TIMESTEP_SHIFT - 1) * t)   # shift in (0,1)
        tb = t.view(B, 1, 1, 1, 1)
        xt = (1 - tb) * x0 + tb * noise
        v_target = noise - x0

        # 33-channel DiT input per item
        mask = torch.zeros(B, 1, F_CROP, h, w, device=device, dtype=torch.bfloat16)
        mask[:, :, 1:] = 1.0
        dit_in = [make_dit_input(xt[i], ref[i], mask[i]) for i in range(B)]

        ctx = [torch.zeros(TEXT_LEN, TEXT_DIM, device=device, dtype=torch.bfloat16)
               for _ in range(B)]
        t_frames = (t.view(B, 1) * 1000).expand(B, F_CROP)   # student wants [B,F]
        seq_len = F_CROP * h * w // (1 * 2 * 2)

        pred = model(dit_in, t_frames, ctx, seq_len, audio_emb=audio)  # [B,16,F,h,w]
        loss = F.mse_loss(pred.float(), v_target.float())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(loss.item())

        if step < 3 or step % args.log_every == 0 or step == args.steps - 1:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  step {step:4d}  loss={loss.item():.4f}  "
                  f"peak_mem={peak:.1f}GB", flush=True)

    dt = time.time() - t_start
    n = max(1, len(losses) // 10)
    print(f"\ntrained {args.steps} steps in {dt/60:.1f} min ({dt/args.steps:.2f}s/step)")
    print(f"  loss: first-{n} avg={sum(losses[:n])/n:.4f}  "
          f"last-{n} avg={sum(losses[-n:])/n:.4f}")
    print(f"  has_nan={any(l != l for l in losses)}")

    out = os.path.join(args.data_dir, "stage1_ckpt.pt")
    torch.save({n: p for n, p in model.named_parameters() if p.requires_grad}, out)
    print(f"  saved trainable params -> {out}")


if __name__ == "__main__":
    main()
