# RNJ-1-Instruct

8B parameter code and STEM model by Essential AI.

## Overview

RNJ-1-Instruct is a dense language model trained from scratch, optimized for:
- **Code generation**: Strong on HumanEval+, MBPP+, LiveCodeBench
- **Agentic tasks**: 20.8% on SWE-bench Verified (bash-only mode)
- **Tool calling**: Top performer on Berkeley Functional Calling Leaderboard
- **Math/STEM**: Strong on GSM8k, Minerva-MATH, AIME

## Specifications

| Parameter | Value |
|-----------|-------|
| Parameters | 8.3B |
| Context Length | 32K |
| License | Apache 2.0 |
| VRAM Required | ~18GB (BF16) |

## Usage

### Build

```bash
cvl run rnj-1-instruct build
# or
./build.sh
```

### Run Inference

```bash
# Default prompt
cvl run rnj-1-instruct predict

# Custom prompt
cvl run rnj-1-instruct predict -- --prompt "Write a Python function to sort a list"

# With options
cvl run rnj-1-instruct predict -- \
  --prompt "Implement binary search" \
  --temperature 0.5 \
  --max-tokens 1024 \
  --format json
```

### Smoke Test

```bash
cvl run rnj-1-instruct test
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | (palindrome example) | Input prompt |
| `--system` | "You are a helpful assistant." | System prompt |
| `--max-tokens` | 512 | Maximum tokens to generate |
| `--temperature` | 0.2 | Sampling temperature (0 for greedy) |
| `--top-p` | 0.95 | Top-p sampling parameter |
| `--format` | txt | Output format (txt/json) |
| `--output` | outputs/result.{ext} | Output file path |
| `--device` | auto | Device (cuda/mps/cpu) |
| `--verbose` | false | Enable verbose logging |

## Notes

- The model is optimized for code and STEM tasks, not general factual knowledge
- Use temperature 0-0.6 for best results
- Include a system prompt to prevent excessive code generation in non-code contexts

## References

- Model Card: https://huggingface.co/EssentialAI/rnj-1-instruct
- Announcement: https://essential.ai/research/rnj-1
