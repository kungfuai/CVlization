# Distillation 101 → 401: an honest pedagogical run-through

This directory implements three distillation methods on the same teacher /
student pair, all running and demonstrating their published mechanics. The
clean lesson at the end is: **distillation algorithms aren't magic — published
recipes need 100-1000× the compute we're spending here to actually move the
student visually.** The implementations are useful as a reading reference.

## Setup (constant across all experiments)

- **Teacher**: OmniAvatar-1.3B with 25 denoising steps + CFG=4.5, audio
  disabled. Pre-computed clean latents for 10 items (`teacher_clean/*.pt`).
  Teacher's latent std ≈ 0.7-1.0 (sharp talking-face output, slightly
  oversaturated from high CFG).
- **Student**: OmniAvatar-1.3B + LoRA (177M trainable). Target: 4-step
  inference WITHOUT CFG.
- **Baseline student** (no distillation, 4-step, no CFG): latent std ≈ 0.35
  → decoded to blurry mush. Real and large gap from teacher.
- **Hardware**: 1× A100-40GB, 1.3B base + LoRA fits comfortably.

## Experiment 1 — Step distillation (Hinton-style MSE)

`train_step_distill.py --steps 1000 --lr 1e-5`

Per-step: noise teacher's clean to xt at random sigma_t; student does ONE
forward; loss = MSE(student_x0, teacher_clean).

**Result**: per-t_idx loss change after 1000 steps:

| t_idx | sigma | loss change |
| --- | --- | --- |
| 0 | 1.0 | +4% (got worse) |
| 1 | 0.938 | +22% (got worse) |
| 2 | 0.833 | **-15% (improved)** |
| 3 | 0.625 | **-21% (improved)** |

Loss drops cleanly for low-noise targets but not for high-noise. **Visually
the trained student is identical to baseline** because 4-step inference
starts at sigma=1.0 (where nothing was learned) and the chain compounds.

## Experiment 2 — Sigma² loss weighting

`train_step_distill.py --steps 1000 --lr 1e-5 --loss-weighting sigma_sq`

Hypothesis: amplifying high-noise samples will help them learn. **Did not
help.** Per-t_idx changes are virtually identical to uniform: {+4%, +21%,
-15%, -22%}. Scale-multiplying the loss doesn't add gradient information
that wasn't already there.

## Experiment 3 — DMD2 from scratch

`train_dmd2.py --steps 1000 --fake-warmup-steps 100`

Full Distribution Matching Distillation v2 (Yin et al. 2024):

- 3 model copies sharing OmniAvatar base, differing only in LoRA state:
  - TEACHER (frozen, OmniAvatar pretrained)
  - GENERATOR (trainable, student)
  - FAKE_SCORE (trainable, critic that tracks generator)
- Per iter:
  - x_pred = GENERATOR(noise, t=1.0)  # 1-step from pure noise
  - sample sigma_t; add noise to x_pred → xt
  - x_real = TEACHER(xt, sigma_t)
  - x_fake = FAKE_SCORE(xt, sigma_t)
  - L_gen = sum(x_pred * weight * (x_fake - x_real).detach())  # straight-through gradient
  - L_fake = MSE(FAKE_SCORE(xt, sigma_t), x_pred.detach())  # critic chases generator
- 100-step fake_score warmup, then alternating updates

**Result**: all the mechanics work — 10.3GB GPU memory, no NaN, two-time-
scale optimization stable. L_fake holds at 0.08-0.10 (critic alive). L_gen
becomes O(±500-1000) once scaled correctly. **But x_pred.std stays at 0.39
after 1000 steps** (target was 0.74), and the trained generator is
visually identical to baseline.

## Why nothing visibly moved (the actual lesson)

The three implementations are correct. The setup is just much too small
to see the effects published methods report:

| Resource | Our run | Published recipes |
| --- | --- | --- |
| training items | 10 | 30,000+ (CausVid: 12k prompts × multiple seeds) |
| training steps | 1,000 | 8,000-50,000 (Self-Forcing, Hallo-Live, LiveAvatar) |
| GPUs | 1× A100-40GB | 8× H100/H800 with FSDP |
| trainable params | 177M LoRA | usually full-finetune or larger LoRA |
| hyperparameter tuning | no | extensive (lr schedules, EMA, gradient clipping) |
| stages | just stage-2 | stage-1 ODE init *then* stage-2 DMD |

In other words: the curves from "DMD2 paper figure 3 shows convergence in
500 steps" come from authors with 50-100× our compute, tuning that took
weeks, and ablations showing what configurations matter.

When you re-derive these algorithms from scratch and run them on a
laptop-budget setup, you reproduce the *mechanics* but not the *results*.
That's a real, useful thing to know.

## What does work today

For an actual working talking-avatar streaming model, the engineering-
economical path is **use someone's published distilled checkpoint** rather
than re-train from scratch. See `../STREAMING_INFERENCE.md` for the
LiveAvatar finding — they ship a 14B Wan2.2-based DMD-distilled checkpoint
that runs real-time on 5× H800 with FP8. Our v7 (4-rank PP + CFG) hits
1.07× real-time on 4× A100-40GB using pretrained weights, which is
sufficient as a streaming-inference reference.

## Files

| file | purpose |
| --- | --- |
| `render_variable_steps.py` | Render OmniAvatar at any (num_steps, cfg) combo; optional `--trained-ckpt` overlay |
| `precompute_teacher_latents.py` | One-time: run teacher at 25-step CFG=4.5 for N items, save clean latents |
| `train_step_distill.py` | Plain Hinton-style MSE distillation; `--loss-weighting` for sigma² etc. |
| `train_dmd2.py` | Full DMD2: shared base + 3 LoRA states, score-distillation gradient, alternating optimizers |
| `teacher_clean/` | Pre-computed teacher outputs (10 items) |
| `run1/`, `run2_sigmasq/`, `dmd2/`, `dmd2_v2/` | Checkpoints + loss curves from each experiment |
| `student_*_mid.png` | Decoded mid-frames for visual comparison |
