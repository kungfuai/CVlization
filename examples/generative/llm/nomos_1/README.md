# Nomos-1: Mathematical Reasoning Model

Nomos-1 is a 31B parameter Mixture-of-Experts model (~3B active parameters) specialized for mathematical problem-solving and proof-writing. Fine-tuned from Qwen3-30B-A3B-Thinking by Nous Research.

## Features

- Mathematical reasoning and proof generation
- 87/120 on Putnam 2025 benchmark (vs 24/120 for base model)
- Efficient MoE architecture (only ~3B params active per forward pass)
- Apache 2.0 license

## Requirements

- **GPU**: 1x with 64GB+ VRAM recommended (tested on 96GB)
- **Disk**: ~80GB for model weights
- **Docker**: Required

## Quick Start

```bash
# Build the Docker image
./build.sh

# Run inference with default math prompt
./predict.sh

# Custom math problem
./predict.sh --prompt "Prove that the square root of 2 is irrational"

# Smoke test
./test.sh
```

## Usage

```bash
# Basic usage
./predict.sh --prompt "Solve x^2 - 5x + 6 = 0"

# JSON output format
./predict.sh --prompt "What is the sum of 1+2+...+100?" --format json

# Adjust generation parameters
./predict.sh --prompt "Prove there are infinitely many primes" \
  --max_new_tokens 4096 \
  --temperature 0.3

# Greedy decoding (deterministic)
./predict.sh --prompt "Calculate 17 * 23" --greedy
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | "Prove that there are infinitely many prime numbers." | Math problem or question |
| `--output` | `outputs/result.{format}` | Output file path |
| `--format` | `txt` | Output format (`txt` or `json`) |
| `--max_new_tokens` | 2048 | Maximum tokens to generate |
| `--temperature` | 0.6 | Sampling temperature |
| `--top_p` | 0.95 | Top-p (nucleus) sampling |
| `--greedy` | false | Use greedy decoding |
| `--verbose` | false | Enable verbose logging |

## Example Problems

```bash
# Number theory
./predict.sh --prompt "Prove that there are infinitely many prime numbers"

# Algebra
./predict.sh --prompt "Solve the system: 2x + 3y = 7, x - y = 1"

# Calculus
./predict.sh --prompt "Find the derivative of f(x) = x^3 * ln(x)"

# Combinatorics
./predict.sh --prompt "How many ways can 8 rooks be placed on a chessboard so that no two attack each other?"
```

## Notes

- First run downloads ~80GB model weights to `~/.cache/huggingface`
- Model uses bfloat16 on modern GPUs (compute capability >= 8)
- Blackwell GPUs (RTX 50xx, B100, etc.) require PyTorch 2.9.1+ with CUDA 12.8
- For optimal performance on competition-level math, consider the [Nomos Reasoning Harness](https://github.com/NousResearch/nomos)

## References

- Model: https://huggingface.co/NousResearch/nomos-1
- Nomos Reasoning Harness: https://github.com/NousResearch/nomos
- Base Model: https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507
