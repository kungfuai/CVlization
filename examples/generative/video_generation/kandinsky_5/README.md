# Kandinsky 5.0 Video/Image Generation

Kandinsky 5.0 is a family of diffusion models for video and image generation from Sber AI.

- **Video Lite**: 2B parameter T2V/I2V model, runs on 12GB+ VRAM
- **Distilled**: 16-step variant for 6x faster inference
- **Multi-lingual**: Supports English and Russian prompts

## Requirements

- NVIDIA GPU with 12GB+ VRAM (24GB recommended)
- 32GB+ system RAM (or 16GB with Qwen quantization)
- ~25GB disk space for model weights

## Setup

### 1. Build the Docker image

```bash
./build.sh
```

## Quick Start

Models auto-download on first run (~25GB to `~/.cache/cvlization/kandinsky5`).

```bash
# Text-to-Video (5 seconds)
./predict.sh --prompt "A cat playing piano"

# With optimizations for low VRAM
./predict.sh --prompt "A dog running" --offload --qwen_quantization

# Distilled model (6x faster)
./predict.sh --prompt "Ocean waves" --config distilled

# Text-to-Image
./predict.sh --prompt "A red hat" --mode t2i
```

## Options

- `--prompt`: Text prompt for generation
- `--config`: Model variant (sft, distilled, nocfg)
- `--mode`: Generation mode (t2v, i2v, t2i)
- `--duration`: Video duration in seconds (5 or 10)
- `--offload`: Enable CPU offloading for low VRAM
- `--qwen_quantization`: Use 4-bit quantized Qwen encoder
- `--output`: Output file path

## Model Variants

| Model | Steps | Speed | Quality |
|-------|-------|-------|---------|
| SFT | 50 | 1x | Best |
| NoCFG | 50 | 2x | Good |
| Distilled | 16 | 6x | Good |

## License

MIT License. See [LICENSE](https://github.com/kandinskylab/kandinsky-5/blob/main/LICENSE).
