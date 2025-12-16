# Miles GRPO Training with Qwen3-0.6B

This example demonstrates [Miles](https://github.com/radixark/miles), an enterprise-grade reinforcement learning framework for large-scale LLM post-training. Miles is forked from [slime](https://github.com/THUDM/slime) and provides:

- **Decoupled architecture**: Megatron/FSDP for training, SGLang for inference
- **True on-policy RL**: Zero mismatch between training and inference
- **Production features**: Fault tolerance, memory robustness, speculative training

This example uses **Qwen3-0.6B** to enable single-GPU training on consumer hardware (e.g., RTX 3090 24GB).

## Model Choice

We chose Qwen3-0.6B because:
- Small enough for single GPU (0.6B params, ~1.2GB in bf16)
- Miles has built-in support via `scripts/models/qwen3-0.6B.sh`
- Same architecture as larger Qwen3 models used in production Miles deployments

### Why Not Nanochat?

We considered [nanochat](https://huggingface.co/karpathy/nanochat-d32) (~560M params) as an alternative:
- Pros: Lightweight, well-documented training from scratch
- Cons: Miles requires an `mbridge` converter for weight synchronization between Megatron and SGLang. Nanochat uses a custom architecture that would need a new converter implementation.

Adding nanochat support would require implementing a new bridge in `miles_plugins/mbridge/nanochat.py`. Contributions welcome!

## Hardware Requirements

| Configuration | GPUs | VRAM | Notes |
|---------------|------|------|-------|
| **Minimum** | 1 GPU | 24GB | Qwen3-0.6B with FSDP + colocate |
| **Recommended** | 2 GPUs | 48GB+ | Better throughput |
| **Production** | 4+ GPUs | 80GB+ | Full Miles features |

## Quick Start

### 1. Build the Docker image

```bash
./build.sh
```

### 2. Run training

```bash
./train.sh
```

### 3. Smoke test

```bash
./test.sh
```

## Configuration

Edit `config.yaml` to customize training:

```yaml
model:
  name: "Qwen/Qwen3-0.6B"

training:
  num_rollout: 10        # Increase for full training
  rollout_batch_size: 4  # Reduce if OOM
  global_batch_size: 16
  learning_rate: 1e-6
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Miles Framework                          │
├─────────────────────────────────────────────────────────────┤
│  Training (FSDP)          Data Buffer         Rollout (SGLang) │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐ │
│  │ Gradient      │◄───│ Prompts +     │◄───│ Generate      │ │
│  │ Updates       │    │ Rewards       │    │ Samples       │ │
│  └───────────────┘    └───────────────┘    └───────────────┘ │
│         │                                          ▲          │
│         └──────────── Weight Sync ─────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

With `--colocate` mode, training and inference share the same GPUs, enabling single-GPU operation.

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--train-backend` | `fsdp` or `megatron` | `fsdp` |
| `--colocate` | Share GPUs for train+inference | enabled |
| `--advantage-estimator` | `grpo`, `ppo`, `reinforce_plus_plus` | `grpo` |
| `--rollout-batch-size` | Samples per rollout | 4 |
| `--global-batch-size` | Total batch size | 16 |

## Output

Training outputs are saved to `outputs/`:
- `checkpoints/` - Model checkpoints
- `logs/` - Training logs
- `wandb/` - W&B artifacts (if enabled)

## Troubleshooting

### OOM Errors

Reduce batch sizes in `config.yaml`:
```yaml
training:
  rollout_batch_size: 2
  global_batch_size: 8
  max_tokens_per_gpu: 4096
```

### Ray Initialization Errors

Ensure no other Ray processes are running:
```bash
ray stop --force
pkill -9 ray
```

## References

- [Miles GitHub](https://github.com/radixark/miles)
- [Miles FSDP Blog](https://lmsys.org/blog/2025-12-03-miles-fsdp/)
- [Qwen3 Models](https://huggingface.co/Qwen)
- [slime (upstream)](https://github.com/THUDM/slime)
