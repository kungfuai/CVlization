# tiny-dllm

Character-level diffusion and GPT language models (~10.7M params) trained on Tiny Shakespeare. Based on [nathan-barry/tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion).

Two modes:
- **diffusion**: bidirectional attention, confidence-based parallel decoding
- **gpt**: causal attention, autoregressive decoding

## Quickstart

```bash
# Build
cvl run tiny-dllm build

# Train diffusion model (~10min)
cvl run tiny-dllm train

# Train GPT model
cvl run tiny-dllm train -- --model gpt

# Generate text
cvl run tiny-dllm predict
cvl run tiny-dllm predict -- --model gpt

# Generate with custom prompt
cvl run tiny-dllm predict -- --prompt "ROMEO:" --max-tokens 500
```

## Arguments

### train.py
| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `diffusion` | `diffusion` or `gpt` |
| `--iters` | `10000` | Training iterations |
| `--batch-size` | `64` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--eval-interval` | `500` | Eval frequency |

### predict.py
| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `diffusion` | `diffusion` or `gpt` |
| `--max-tokens` | `2000` | Characters to generate |
| `--temperature` | `0.8` | Sampling temperature |
| `--confidence` | `0.95` | Diffusion decode threshold |
| `--top-k` | `2` | Diffusion top-k sampling |
| `--prompt` | *(from data)* | Seed text |

## References

- Original repo: [nathan-barry/tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion)
- Inspired by Karpathy's [nanochat](https://github.com/karpathy/nanochat) and ["Let's build GPT"](https://github.com/karpathy/ng-video-lecture)
