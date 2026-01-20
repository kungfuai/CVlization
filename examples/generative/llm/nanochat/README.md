# nanochat - Dockerized

Dockerized version of Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) - a full-stack ChatGPT implementation in a single, clean, minimal codebase.

## Overview

nanochat trains a ChatGPT-like LLM from scratch including:
- Tokenization
- Pretraining
- Finetuning
- Evaluation
- Inference
- Web UI for chatting

The original $100 "speedrun" trains a 1.9B parameter model (d32) in ~4 hours on 8xH100 GPUs.

## Quick Start

### 1. Build the Docker Image

```bash
bash build.sh
```

This builds the container using the vendored nanochat dependency metadata in `examples/generative/llm/nanochat/nanochat`.
The source code itself is mounted at runtime.

### 2. Run Training (Single GPU)

The `train.sh` script supports all 4 training stages:

```bash
# Base pretraining (default)
bash train.sh base --depth=8 --num_iterations=100

# Midtraining on curated data
bash train.sh mid --depth=8

# Supervised fine-tuning for chat
bash train.sh sft --depth=8

# Reinforcement learning
bash train.sh rl --depth=8

# Run full pipeline (base -> mid -> sft -> rl)
bash train.sh all --depth=4
```

For quick testing, run base training with minimal iterations:

```bash
bash train.sh base --depth=4 --num_iterations=20
```

### 3. Run Speedrun (8xH100)

If you have access to an 8xH100 node:

```bash
bash speedrun.sh
```

This runs the full $100 speedrun script that trains a 1.9B parameter model.

### 4. Chat with Your Model

After training completes:

```bash
bash chat_web.sh
```

Then visit http://localhost:8000 to chat with your trained model.

## Single GPU Training

The speedrun requires 8xH100 GPUs, but you can train smaller models on a single GPU:

```bash
# Interactive shell
docker run --rm -it --gpus all \
    -v $(pwd):/workspace \
    nanochat bash

# Inside container
cd /workspace/nanochat
source .venv/bin/activate

# Train tiny model (fits on most GPUs)
python -m scripts.base_train --depth=8 --device_batch_size=4
```

## Configuration

Key parameters to adjust for your GPU:

- `--depth`: Model depth (8, 16, 26, 32). Higher = more parameters
- `--device_batch_size`: Batch size per GPU (reduce if OOM)
- Reduce both to fit smaller GPUs
- Example: `--depth=8 --device_batch_size=2` for 16GB VRAM

## Scripts

- `build.sh` - Build Docker image
- `train.sh` - Train on single GPU (supports base/mid/sft/rl/all modes)
- `speedrun.sh` - Full 8xH100 speedrun
- `chat_web.sh` - Launch web UI
- `shell.sh` - Interactive container shell
- `test.sh` - Verify Docker setup

## Architecture

nanochat is a full implementation including:

- **Tokenizer**: BPE tokenizer (rustbpe)
- **Model**: Transformer with configurable depth
- **Training**: Base pretraining → Midtraining → SFT → RL
- **Evaluation**: CORE, ARC, GSM8K, HumanEval, MMLU benchmarks
- **Serving**: FastAPI web server with ChatGPT-like UI

## Resource Requirements

| Model | Depth | Params | VRAM | Time (8xH100) | Cost |
|-------|-------|--------|------|---------------|------|
| Speedrun | 32 | 1.9B | 80GB | ~4 hours | ~$100 |
| Medium | 26 | ~1.5B | 80GB | ~12 hours | ~$300 |
| Small | 16 | ~800M | 40GB | ~2 hours | ~$50 |
| Tiny | 8 | ~200M | 16GB | ~30 min | ~$12 |

## Performance

The $100 speedrun (d32, 1.9B params) achieves:

| Metric | Score |
|--------|-------|
| CORE | 0.2219 |
| ARC-Challenge | 0.2875 |
| ARC-Easy | 0.3561 |
| GSM8K | 0.0250 |
| HumanEval | 0.0671 |
| MMLU | 0.3111 |

This outperforms GPT-2 (2019) but falls short of modern LLMs.

## TODO

- [ ] Flash Attention 3 support for Blackwell (sm100) and future architectures (sm120). Currently falls back to PyTorch SDPA on non-Hopper GPUs.

## Reference

- Original repo: https://github.com/karpathy/nanochat
- Inspired by: [nanoGPT](https://github.com/karpathy/nanoGPT)
- Course: [LLM101n](https://github.com/karpathy/LLM101n) by Eureka Labs

## License

MIT (matches upstream nanochat)
