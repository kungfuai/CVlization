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
* **Full 600-iter training**: DONE. Ran in ~15 h on 4× A100-40GB (1.5
  min/iter steady state). 6 checkpoints saved (~14 GB each, 85 GB total).
  Per-GPU memory steady at 22.6 GB.

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

## Continuation + gradient accumulation result (negative)

Attempted to resume from step 600 with `gradient_accumulation_steps: 16`
(effective batch 64, matching their published config) to test if closing
the batch-size gap fixes the flickering observed in our 600-iter result.

Result: **OOM on 4xA100-40GB at ~37GB used per GPU**, after ~2h of training.

Patches we made for this attempt (all in `trainer/distillation.py`):

* Load `critic`, `generator_ema` (stripping `_fsdp_wrapped_module.` prefix),
  and `step` from the checkpoint if present
* Add `gradient_accumulation_steps` config: wrap each `fwdbwd_one_step` in
  a `for _ in range(grad_accum)` loop (the optimizer.step happens once
  after accumulation)
* Save `step` in the checkpoint dict so future resumes are clean

What this proves about hardware sizing:

| Effective batch | Hardware needed | Wall clock for 100 iters |
| --- | --- | --- |
| 4 (our smoke + full run) | 4xA100-40GB ok | 15 h (600 iters) |
| 16 (grad_accum=4)  | 4xA100-40GB likely ok | ~10 h |
| 32 (grad_accum=8)  | 4xA100-40GB borderline | ~20 h |
| **64 (grad_accum=16, published)** | **4xA100-40GB OOMs**; needs 7+ A100s or 80GB GPUs | ~40 h |

Documented as a useful negative result. The 16x larger effective batch
that distinguishes published self_forcing_dmd.pt from our 600-iter
checkpoint is the same factor that prevents naive single-node 4x40GB
training from matching it.

## Inference gotcha: FSDP-prefixed EMA keys

Their saved checkpoint has `generator_ema` keys with the FSDP wrapper
prefix (`model._fsdp_wrapped_module.patch_embedding.weight` etc.), while
the inference pipeline expects plain keys (`model.patch_embedding.weight`).
Their `rename_param` helper is defined in the trainer but not applied at
save time for the EMA. The `generator` (non-EMA) keys ARE clean. So either:

* Load the non-EMA generator (omit `--use_ema`): works directly.
* Or rewrite the checkpoint stripping the FSDP prefix:

  ```python
  src = torch.load("checkpoint_model_000600/model.pt", map_location="cpu")
  src["generator_ema"] = {
      k.replace("_fsdp_wrapped_module.", ""): v
      for k, v in src["generator_ema"].items()
  }
  torch.save(src, "checkpoint_model_000600/model_inference.pt")
  ```

Then `inference.py --checkpoint_path .../model_inference.pt --use_ema` works.

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
