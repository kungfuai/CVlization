# TurboDiffusion

Fast text-to-video generation using TurboDiffusion, achieving **100x+ speedup** over standard diffusion models.

## Overview

TurboDiffusion accelerates Wan2.1 video generation through:
- **RCM Distillation**: Reduces sampling steps from 50+ to just 4
- **SageSLA Attention**: Sparse attention with 8-bit quantization
- **Int8 Quantization**: Reduces model memory footprint

This example uses the `TurboWan2.1-T2V-1.3B-480P` model (quantized) which generates 5-second 480p videos in ~2 seconds on RTX 4090/5090.

## Requirements

- NVIDIA GPU with 24GB+ VRAM (RTX 3090/4090/5090, A100, H100)
- Docker with NVIDIA runtime
- ~8GB disk space for model weights

## Quick Start

```bash
# Build the Docker image (compiles CUDA kernels, ~15 min)
./build.sh

# Run smoke test
./test.sh

# Generate custom video
./predict.sh --prompt "A panda eating bamboo in a bamboo forest" --output outputs/panda.mp4
```

## Usage

```bash
./predict.sh [OPTIONS]

Options:
  --prompt TEXT       Text description of the video (default: cat playing piano)
  --output PATH       Output video path (default: outputs/generated_video.mp4)
  --num_steps 1-4     Sampling steps, more = better quality (default: 4)
  --resolution        480p or 720p (default: 480p)
  --seed N            Random seed for reproducibility (default: 42)
```

## Examples

```bash
# High quality (4 steps)
./predict.sh --prompt "A drone shot flying over mountains at sunrise" --num_steps 4

# Fast preview (2 steps)
./predict.sh --prompt "Waves crashing on a rocky shore" --num_steps 2

# 720p output (needs more VRAM)
./predict.sh --prompt "City skyline time-lapse" --resolution 720p
```

## Model Details

| Component | Details |
|-----------|---------|
| Base Model | Wan2.1-T2V-1.3B |
| Checkpoint | TurboWan2.1-T2V-1.3B-480P-quant |
| Parameters | 1.3B |
| Output | 77 frames @ 16fps (~5 seconds) |
| Resolution | 480p (854x480) default |
| VRAM Usage | ~15-18GB (quantized) |

## Performance

On RTX 4090/5090 with quantized model:
- 4 steps: ~2 seconds (E2E diffusion time)
- Full pipeline including text encoding + VAE decode: ~10-15 seconds

## References

- [TurboDiffusion Paper](https://jt-zhang.github.io/files/TurboDiffusion_Technical_Report.pdf)
- [GitHub Repository](https://github.com/thu-ml/TurboDiffusion)
- [HuggingFace Models](https://huggingface.co/TurboDiffusion)

## License

Apache 2.0 (inherits from TurboDiffusion)
