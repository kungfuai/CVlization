# Self-Forcing DMD training on 4× A100-40GB

A reproducible recipe for running the official
[Self-Forcing](https://github.com/guandeh17/Self-Forcing) DMD distillation
training on 4 A100-40GB GPUs (instead of their published 8× H100-80GB
single-node or 64× H100 multi-node setup).

Output: a Wan-1.3B text-to-video model whose 4-step inference matches a
50-step teacher in quality, trained via Distribution Matching Distillation
(Yin et al. 2024).

## What this is

The companion to `distill2/` (which contains our from-scratch DMD attempts
that mechanically worked but didn't visibly improve the student). This
directory documents how to run someone else's *proven* DMD recipe on
hardware that's half the size they targeted.

## Status

* **Inference verified**: their released `self_forcing_dmd.pt` produces a
  high-quality 5-sec 832×480 video in ~15 sec on a single A100 (using
  `--use_ema` flag).
* **Training smoke verified**: 50-iter smoke completed on 4 A100-40GB.
  All mechanics work end-to-end; memory ~33.8 GB peak per GPU; no NaN.
  Saved checkpoints at step 25 + 50 (~11 GB each).
* **Full 600-iter training**: not yet run. Estimated ~15 h compute.

## Hardware adaptations from the published recipe

Self-Forcing's published DMD config (`configs/self_forcing_dmd.yaml`)
targets **64× H100-80GB**. To fit on 4× A100-40GB we made these changes,
all in `config_smoke_4gpu.yaml`:

| Field | Published | Ours | Reason |
|---|---|---|---|
| `real_name` | `Wan2.1-T2V-14B` | `Wan2.1-T2V-1.3B` | avoid 28 GB Wan-14B download; self-distillation works for the pedagogical lesson |
| `image_or_video_shape` | `[1, 21, 16, 60, 104]` (480×832 video) | `[1, 21, 16, 30, 52]` (240×416 video) | half resolution to fit activations in 40 GB |
| `total_batch_size` | 64 | 4 | published needs 64 GPUs; we have 4. Their code does NOT do gradient accumulation, so effective batch = world_size |
| `text_encoder_cpu_offload` | (not set) | `true` | save ~5 GB of T5 weights per GPU |
| `max_steps` | (no limit; runs forever) | 50 (smoke) / 600 (full) | for finite training; requires the trainer patch below |

Plus environment + setup:

* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — necessary for
  borderline memory; reduces fragmentation
* Python packages NOT in their `requirements.txt` but needed:
  `lmdb` (dataset), `av` (video writing), `wandb` (imported at top of
  train.py even when `--disable-wandb`)
* Trainer patch: insert `max_steps` early-exit in the `while True:` loop
  of `trainer/distillation.py`:

  ```python
  start_step = self.step
  max_steps = getattr(self.config, "max_steps", 10**9)
  while True:
      if self.step >= max_steps:
          print(f"[max_steps] reached {self.step}; exiting"); break
      ...
  ```

## What "effective batch = 4" means for quality

The published recipe relies on effective batch 64 (= 64 GPUs × batch 1
per GPU, no accumulation). On our 4 GPUs we get batch 4 — 16× smaller.
This will likely:
* Slow convergence (more iters needed for same loss reduction)
* Or hurt final quality
* Or just work fine — depends on how robust their setup is

To match published quality, you'd add gradient accumulation to the
trainer (~30 lines of code change) to do 16 forward+backward passes
before each `optimizer.step()`. That increases per-iter wall-clock 16×.

For "learn distillation by watching it run end-to-end" the smaller
effective batch is fine. For "match published quality" it isn't.

## Reproducing the smoke (on a fresh box)

```bash
# 1) Clone the upstream repo (fresh, no patches)
git clone https://github.com/guandeh17/Self-Forcing.git /tmp/cvl
cd /tmp/cvl

# 2) Use any existing Wan-1.3B install or download:
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir wan_models/Wan2.1-T2V-1.3B

# 3) Download Self-Forcing's ODE-init checkpoint + text prompts
huggingface-cli download gdhe17/Self-Forcing \
    checkpoints/ode_init.pt vidprom_filtered_extended.txt \
    --local-dir .

# 4) Install missing deps
pip install lmdb av wandb

# 5) Apply the max_steps patch (see README above)

# 6) Drop in the smoke config + run
cp /path/to/this/config_smoke_4gpu.yaml configs/smoke.yaml
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --master_port=29250 train.py \
    --config_path configs/smoke.yaml \
    --logdir logs/smoke --disable-wandb --no_visualize
```

Smoke completes in ~75 min on 4× A100-40GB. Two checkpoints (step 25 +
step 50) saved as `logs/smoke/checkpoint_model_NNNNNN/model.pt`.

## What to look at after smoke runs

* `logs/smoke/checkpoint_model_000050/model.pt` — the trained generator
  state_dict. Load with their `inference.py --use_ema` flag (per their
  README) to render a video and compare to baseline.
* GPU memory should stay under 40 GB throughout. If you see OOM, halve
  the spatial dimensions further or use 7-8 GPUs.

## Files in this directory

| file | purpose |
| --- | --- |
| `README.md` | this file |
| `config_smoke_4gpu.yaml` | the YAML config we used for the smoke; drop into `configs/` of a Self-Forcing checkout |

## Not in this directory (but referenced)

| | |
|---|---|
| max_steps patch | inline above; small edit to `trainer/distillation.py` |
| Self-Forcing repo | external: https://github.com/guandeh17/Self-Forcing |
| Their distilled checkpoint | external: `huggingface.co/gdhe17/Self-Forcing` (5.68 GB) |
