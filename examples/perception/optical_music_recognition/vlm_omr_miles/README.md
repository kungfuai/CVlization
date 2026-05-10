# Miles GRPO for VLM OMR

RL post-training for Optical Music Recognition using [Miles](https://github.com/radixark/miles) — a decoupled RL framework with parallel rollouts via SGLang.

## Why Miles over TRL/unsloth GRPO

| Feature | TRL GRPO | Miles GRPO |
|---------|----------|------------|
| Rollout | Sequential (in-process vLLM) | **Parallel (SGLang, separate process)** |
| Speed | ~3 min/step with 8 gens | **Parallel generation + training** |
| VLM support | Partial (vision LoRA issues) | **First-class (Qwen3-VL validated)** |
| Vision encoder | Needs merge workaround | **Frozen by design (correct for RL)** |
| On-policy | Approximate | **True on-policy (bit-wise identical)** |

## Quick Start

```bash
./build.sh                    # build Docker image
./train.sh                    # run GRPO training
./train.sh --dry-run          # print Miles command without running
```

## Reward Function

`reward.py` uses `SequenceMatcher.ratio()` on pitched-only sequences —
the exact same computation as `eval_mxc.py`'s `pitched_only_similarity`.

## Configuration

Edit `config.yaml`:
- `model.name`: Qwen3-VL-8B-Instruct (or other Qwen-VL)
- `dataset`: synthetic Level 9 (MXC2 format)
- `training.n_samples_per_prompt`: 8 generations per image
- `grpo.kl_coef`: 0.5 (KL penalty)
