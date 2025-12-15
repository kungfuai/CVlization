# dLLM - Diffusion Language Models

Generate text using diffusion-based language models instead of autoregressive generation.

## Overview

Unlike autoregressive LLMs (GPT, Llama, etc.) that generate tokens left-to-right, diffusion language models generate text through **iterative denoising** - similar to how image diffusion models work:

1. Start with a sequence of mask tokens
2. Iteratively predict and refine tokens over multiple steps
3. Each step denoises the sequence, replacing masks with predicted tokens
4. After N steps, the final sequence is the generated text

## Available Models

| Model | Parameters | VRAM | Diffusion Type | Best For |
|-------|------------|------|----------------|----------|
| `qwen-bd3lm` | 0.6B | ~2GB | BD3LM (block) | Quick testing, low memory |
| `bert-chat` | 395M | ~1GB | MDLM (masked) | Ultra-fast, lightweight |
| `llada` | 8B | ~16GB | MDLM (masked) | Best quality |

## Usage

### Build

```bash
cvl run dllm build
```

### Run Inference

```bash
# Default model (qwen-bd3lm, smallest)
cvl run dllm predict

# Choose a specific model
cvl run dllm predict -- --model qwen-bd3lm --prompt "What is 2+2?"
cvl run dllm predict -- --model bert-chat --prompt "Hello!"
cvl run dllm predict -- --model llada --prompt "Write a poem about AI"

# With options
cvl run dllm predict -- \
  --model qwen-bd3lm \
  --prompt "Explain neural networks" \
  --steps 256 \
  --max-tokens 512
```

### List Available Models

```bash
cvl run dllm predict -- --list-models
```

### Smoke Test

```bash
cvl run dllm test
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | qwen-bd3lm | Model: qwen-bd3lm, bert-chat, llada |
| `--prompt` | (math question) | Input prompt |
| `--steps` | 128 | Number of diffusion denoising steps |
| `--max-tokens` | 256 | Maximum tokens to generate |
| `--block-size` | 32 | Block size for diffusion |
| `--temperature` | 0.0 | Sampling temperature (0 for greedy) |
| `--format` | txt | Output format (txt/json) |
| `--seed` | 42 | Random seed |
| `--list-models` | - | List available models and exit |

## Diffusion Methods

### BD3LM (Block Discrete Denoising Diffusion)
- Used by: `qwen-bd3lm`
- Generates text in blocks, using KV cache for efficiency
- Paper: https://arxiv.org/abs/2503.09573

### MDLM (Masked Diffusion Language Modeling)
- Used by: `bert-chat`, `llada`
- Global masked diffusion across all tokens
- Paper: https://arxiv.org/abs/2406.07524

## References

- dLLM Library: https://github.com/ZHZisZZ/dllm
- LLaDA Paper: https://arxiv.org/abs/2502.09992
- BD3LM Paper: https://arxiv.org/abs/2503.09573
- MDLM Paper: https://arxiv.org/abs/2406.07524
