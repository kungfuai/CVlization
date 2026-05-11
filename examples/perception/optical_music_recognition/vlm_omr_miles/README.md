# Miles GRPO for VLM OMR

RL post-training pipeline for Optical Music Recognition using [Miles](https://github.com/radixark/miles) — a decoupled RL framework with parallel rollouts via SGLang.

## Status: ~80% working, blocked on GDN kernel limitation

As of 2026-05-11, Qwen3.5-VL training in Miles hits a Megatron-LM
kernel limitation: `NotImplementedError: GDN does not support packed
sequence for now`. See [`reports/2026-05-11-miles-rl.md`](../vlm_omr_sft/reports/2026-05-11-miles-rl.md)
for the full debugging journey and lessons learned.

**Text-only Qwen3.5 works in Miles** (validated by Miles PR #740).
The blocker is specific to the VL variant + GDN attention + packed seqs.

## What works (verified)

- ✅ Megatron model loaded via bridge (with MTP patch)
- ✅ SGLang VLM engine started (TP=2, both GPUs)
- ✅ Rollouts generated (`POST /generate HTTP/1.1 200 OK`)
- ✅ Custom reward computed
- ❌ Training step crashes in GDN attention kernel

## Why this pipeline is kept

If Qwen3.5-VL becomes production-ready in Miles (upstream fixes for
slime #1815, #1894, #1713), this code is ready to launch — just
remove the GDN-related blocker.

Alternative: retrain SFT from text-only Qwen3.5-9B (no vision), then
Miles works today. Trade-off: lose vision capability that the OMR
task fundamentally requires.

## Quick reference

```bash
./build.sh                              # builds image + applies MTP patch
./train.sh                              # uses config.yaml (full dataset)
./train.sh --config config_validate.yaml  # 2 rollouts × 2 gens, ~5 min
```

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | radixark/miles:latest + OMR deps + MTP patch script |
| `qwen3_5_mtp_patch.py` | Patches megatron-bridge for MTP layer rename (slime #1894) |
| `train.py` | Dataset prep + Miles CLI command builder for Qwen3.5-9B Megatron |
| `reward.py` | Async OMR reward (SequenceMatcher on MXC2 pitches) |
| `config.yaml` | Full Level 9 dataset training config |
| `config_validate.yaml` | Minimal end-to-end validation config |
| `test_sglang_load.py` | Standalone SGLang VLM compatibility test |

## Why Miles over TRL/unsloth GRPO (theoretical)

| Feature | TRL GRPO | Miles GRPO |
|---|---|---|
| Rollout | Sequential (in-process vLLM) | Parallel (SGLang separate process) |
| VLM support | Works (with merge step) | Advertised but Qwen3.5-VL not production-ready |
| Vision encoder | Trainable LoRA (we proved unnecessary) | Frozen by design |
| On-policy | Approximate | True on-policy (bit-wise identical) |
| Training type | LoRA (r=32) | Full fine-tune (Megatron) |

For our specific task, TRL works end-to-end (maintained 84% on Level 9
across multiple GRPO runs). Miles would offer more capacity but isn't
yet stable for Qwen3.5-VL.
