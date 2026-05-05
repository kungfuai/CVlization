# TIDE - Cross-Architecture Distillation for Diffusion LLMs

Distill knowledge from a larger diffusion language model (teacher) into a smaller one (student) using the TIDE recipe.

## Overview

TIDE introduces three components for dLLM distillation:

- **TIDAL** (scheduling): cosine-interpolated soft targets that gradually shift from student self-supervision to teacher guidance
- **CompDemo** (context): complementary mask-splitting that gives the teacher ~50% more context per forward pass, improving signal quality at high masking ratios
- **Reverse CALM** (output, Pipeline A only): gradient-stable cross-tokenizer alignment via reversed BCE

This example implements **Pipeline B** (shared-tokenizer distillation with TIDAL + CompDemo). The default setup distills from WeDLM-8B-Instruct into Qwen3-0.6B-BD3LM.

## Usage

### Build

```bash
cvl run tide-dllm-distill build
```

### Train (distillation)

```bash
# Self-distillation (smoke test, ~5GB VRAM)
cvl run tide-dllm-distill train

# With a real teacher (~20GB VRAM)
cvl run tide-dllm-distill train -- \
  --teacher-model WeDLM/WeDLM-8B-Instruct \
  --steps 10000

# Custom dataset
cvl run tide-dllm-distill train -- \
  --dataset tatsu-lab/alpaca \
  --max-length 512 \
  --batch-size 4
```

### Inference

```bash
# Base (undistilled) model
cvl run tide-dllm-distill predict

# TIDE-distilled model (Pipeline B)
cvl run tide-dllm-distill predict -- --model tide-wedlm-shared

# TIDE-distilled model (Pipeline A)
cvl run tide-dllm-distill predict -- --model tide-llada-cross

# Custom prompt
cvl run tide-dllm-distill predict -- \
  --model tide-wedlm-shared \
  --prompt "Write a Python function to check if a number is prime." \
  --steps 256 \
  --max-tokens 512
```

### Smoke test

```bash
cvl run tide-dllm-distill test
```

Runs self-distillation for 10 steps and a quick inference check.

## Available Models

| Key | HuggingFace ID | Description |
|-----|---------------|-------------|
| `base` | `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1` | Base student (undistilled) |
| `tide-wedlm-shared` | `TIDE-dllm/distill-WeDLM-TIDE_Shared` | Pipeline B: TIDAL + CompDemo |
| `tide-llada-cross` | `TIDE-dllm/distill-LLaDA2-TIDE_Cross` | Pipeline A: Reverse CALM |

## VRAM Requirements

| Mode | VRAM |
|------|------|
| Inference (any 0.6B model) | ~2 GB |
| Self-distillation (student = teacher) | ~5 GB |
| Full distillation (8B teacher + 0.6B student) | ~20 GB |

## Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda-init` | 0.1 | TIDAL initial interpolation weight |
| `--lambda-max` | 0.9 | TIDAL final interpolation weight |
| `--temperature` | 2.0 | Softmax temperature for distillation |
| `--ce-weight` | 1.0 | Cross-entropy loss weight |
| `--tidal-weight` | 1.0 | TIDAL distillation loss weight |
| `--use-comp-demo` | true | Enable Complementary Demonstration |
| `--lr` | 5e-5 | Learning rate |
| `--batch-size` | 4 | Batch size |
| `--steps` | 10000 | Training steps |

## References

- Paper: [Turning the Tide: Cross-Architecture Distillation for Diffusion Large Language Models](https://arxiv.org/abs/2604.26951)
- Code: [PKU-YuanGroup/TIDE](https://github.com/PKU-YuanGroup/TIDE)
- Project page: [pku-yuangroup.github.io/TIDE-Page](https://pku-yuangroup.github.io/TIDE-Page/)
