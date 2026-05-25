# Streaming inference for talking avatars — findings

This document summarises the streaming-inference investigation done on top
of `omni_flashtalk`. It captures what works, what doesn't, and the honest
answer for "how do I get a real-time, teacher-quality streaming talking
avatar?".

## What this directory ships (inference/)

Five incrementally-built streaming inference variants, each with verified
behaviour:

| Variant | Compute model | Steady-state throughput on 4× A100-40GB | Output quality |
| --- | --- | --- | --- |
| `streaming_v1_chunked.py` | sequential, 1 GPU | 0.37× real-time | weak (no CFG) |
| `streaming_v2_motion_anchor.py` | sequential + motion-anchor, 1 GPU | 0.41× real-time | weak (no CFG) |
| `streaming_v3_kvcache.py` | KV-cached inference | n/a — broken multi-chunk | broken |
| `streaming_v4_pipeline.py` | single-process 4-GPU PP | 2256 ms/tick (GIL-bound, broken throughput) | weak |
| `streaming_v5_torchrun.py` | torchrun 4-rank multi-process PP | 306-389 ms/tick (matches single-fwd) | weak (no CFG) |
| `streaming_v5_seq.py` | bisect tool: v5 algo on 1 GPU | — | weak (no CFG) |
| `streaming_v6_cfg.py` | CFG-enabled, 1 GPU | 0.24× (4 sec per 0.84 sec) | sharp ✓ |
| `streaming_v7_pp_cfg.py` | PP × 4 GPUs + CFG | **782 ms/tick = 1.07× real-time** | sharp ✓ |

`v7` hits real-time on 4× A100-40GB and matches teacher pixel quality. It
also exposes the two real limitations of our setup:

1. **No cross-chunk attention** (v3 KV cache didn't work for our model;
   see below). So chunks have only motion-frame anchoring as continuity,
   which shows as visible flicker every ~21 pixel frames.
2. **No audio-driven lip motion.** The student is OmniAvatar pretrained,
   never actually distilled to SoulX. Audio passes through the model but
   produces only weak lip movement.

## Why v3 (KV cache) breaks for our model

* Without LoRA injection, KV-cache forward is numerically equivalent to
  no-cache forward on a single chunk (`|out_a - out_b| / std_a ≈ 0.01`,
  within bf16 noise). The cache mechanics in `CausalWanModel` are correct.
* With LoRA loaded from OmniAvatar's pt, **single-chunk** output matches
  the no-cache path. Merging LoRA via `merge_and_unload()` doesn't change
  anything.
* **Multi-chunk** output still degrades (chunk 0 OK, chunks 1+ become
  grey static), regardless of whether LoRA is merged or not.

The deeper cause is that OmniAvatar was never trained for streaming
attention. The cached K/V from prior chunks' clean-context updates
don't represent context the model knows how to attend to. Fixing this
properly requires streaming-aware distillation — i.e., training the
student with KV-cache forward enabled (self-forcing).

## Why we didn't just train the student

We attempted Stage-1 KD (`trainer/train_stage1_ode.py`) using the
Hallo-Live ODE-fusion recipe with teacher trajectories we'd captured
from SoulX. Outcome:

* The pretrained OmniAvatar student is already very close to the
  teacher's output **on pixel quality** when text conditioning is wired
  correctly (initial loss 0.057, decoded mid-frame visually matches
  the teacher in composition and lighting).
* MSE training on raw latents barely moves loss (0.057 → 0.057 over
  hundreds of steps). The gap is in **lip-sync**, which raw-MSE on
  latents is not a useful objective for.
* Proper streaming + lip-sync fix requires Stage-2 DMD with self-rollout,
  reward-weighted dual-stream loss, FSDP for the teacher, and a custom
  trainer. Multi-week effort with real failure risk.

## The found-already-solved answer: LiveAvatar

**LiveAvatar** (Quark-Vision, Dec 2025; arXiv 2512.04677;
`huggingface.co/Quark-Vision/Live-Avatar`) solves this exactly:

* Real-time streaming talking avatar at **45 FPS** on multi-H800
* Distribution-matching distillation (DMD) down to 4 steps
* **Timestep-forcing pipeline parallelism** — literally what our v5/v7
  built independently. Their 4-step model maps 4 ranks one-step-each.
* Block-wise autoregressive with audio-video KV cache
* Built on **Wan2.2-S2V-14B** (speech-to-video DiT) + Wan2.1 VAE +
  `wav2vec2-large-xlsr-53-english` audio encoder
* Released LoRA: `liveavatar.safetensors` (5 GB, 896 keys, rank 128, α 64)

We confirmed the checkpoint is downloadable and the inference scripts
(`run_single.sh`, `infinite_inference_single_gpu.sh`,
`infinite_inference_multi_gpu.sh`) cover both offline and streaming
modes. Hardware caveat: the published configurations target
**single 80 GB GPU** (offline) or **multi-H100/H800 with FP8** (real-time).
On 40 GB A100s we hit OOM in single-GPU mode and a NCCL deadlock in
multi-GPU mode — both look fixable but neither was the goal of this
investigation. The recommendation for any production-quality streaming
talking-avatar work is to use LiveAvatar directly on appropriate
hardware rather than building the trainer from scratch.

## What `omni_flashtalk` is, then

A research demo and an honest engineering investigation:

* `data/` — the SoulX teacher → trajectory → text → audio data
  pipeline (works)
* `trainer/` — the Hallo-Live ODE-fusion KD trainer (works mechanically,
  doesn't bridge the lip-sync gap with this teacher/student pair)
* `inference/` — five streaming inference variants showing how
  pipeline parallelism + CFG combine to hit real-time on 4× A100-40GB.
  Useful as a reference implementation of the LiveAvatar-style streaming
  scheduler, on the Wan-1.3B/OmniAvatar base.

If the goal is a shipping talking-avatar product, the path is **use
LiveAvatar** on 80 GB+ hardware. If the goal is to understand the
moving parts of streaming video distillation on a familiar Wan-1.3B
base, this directory is a working reference.
