# HunyuanVideo-1.5

Video generation model from Tencent with 8.3B parameters, supporting text-to-video (T2V) and image-to-video (I2V) generation.

## Features

- **Text-to-Video (T2V)**: Generate videos from text prompts
- **Image-to-Video (I2V)**: Animate reference images
- **Resolutions**: 480p and 720p (with 1080p super-resolution)
- **Low VRAM**: Runs on 14GB+ GPUs with CPU offloading
- **Speed optimizations**: CFG distillation (2x), step distillation (75% faster), cache inference

## Requirements

- NVIDIA GPU with 14GB+ VRAM (24GB+ recommended)
- **48GB+ system RAM** (required for CPU offloading during model loading)
- CUDA 12.4+
- ~50GB disk space for single variant (full repo is 372GB with all variants)

**Note**: This model requires significant system RAM even with GPU offloading enabled. The Qwen2.5-VL-7B text encoder (~16GB) and transformer checkpoint (~32GB) must be loaded into CPU memory. Systems with less than 48GB RAM may encounter OOM errors during model loading.

## Setup

### 1. Build the Docker image

```bash
./build.sh
```

### 2. Download model weights (~50GB)

Models are downloaded to the standard HuggingFace cache (`~/.cache/huggingface/hub/`).

```bash
./download_models.sh
```

This downloads:
- `tencent/HunyuanVideo-1.5` (480p T2V variant): ~32GB
- `Qwen/Qwen2.5-VL-7B-Instruct`: ~15GB
- `google/byt5-small`: ~1GB
- `black-forest-labs/FLUX.1-Redux-dev` (optional, for I2V): ~1GB

**Note**: Vision encoder requires HF token. Set `HF_TOKEN` in `.env` or environment.

## Quick Start

```bash
# Text-to-video (480p)
./predict.sh --prompt "A cat walking in a garden"

# With CFG distillation (2x faster)
./predict.sh --prompt "Ocean waves at sunset" --cfg_distilled --output outputs/waves.mp4

# Image-to-video (requires vision encoder)
./predict.sh --prompt "The person starts dancing" --image ref.jpg --output outputs/dancing.mp4
```

## Usage

```bash
./predict.sh [OPTIONS]

Required:
  --prompt, -p TEXT       Text prompt for video generation

Optional:
  --image PATH            Reference image for I2V mode
  --output, -o PATH       Output path (default: outputs/output.mp4)
  --resolution {480p,720p}  Video resolution (default: 480p)
  --video_length INT      Number of frames (default: 121, ~5s at 24fps)
  --seed INT              Random seed (default: 42)

Speed Optimizations:
  --cfg_distilled         Enable CFG distillation (~2x speedup)
  --enable_step_distill   Enable step distillation (~75% speedup, 480p I2V only)
  --enable_cache          Enable cache inference (deepcache)
  --use_sageattn          Enable SageAttention

Memory Options:
  --no_offload            Disable CPU offloading (faster but needs more VRAM)
  --no_sr                 Disable super resolution

Multi-GPU:
  --num_gpus INT          Number of GPUs for parallel inference (default: 1)
```

## Model Variants

| Model | Resolution | Task | Speedup |
|-------|------------|------|---------|
| 480p T2V | 480p | Text-to-Video | - |
| 480p I2V | 480p | Image-to-Video | - |
| 480p T2V CFG Distilled | 480p | T2V | 2x |
| 480p I2V CFG Distilled | 480p | I2V | 2x |
| 480p I2V Step Distilled | 480p | I2V | 6x (8-12 steps) |
| 720p T2V | 720p | T2V | - |
| 720p I2V | 720p | I2V | - |

## Performance

On RTX 4090 (24GB):
- 480p I2V with step distillation: ~75 seconds end-to-end
- 480p T2V with CFG distillation: ~3-4 minutes
- 720p T2V: ~8-10 minutes

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--video_length` (default 121 frames)
- Use `--no_sr` to disable super-resolution
- Ensure CPU offloading is enabled (default)

### Model not found
- Run `./download_models.sh` to download all required weights
- Check that `ckpts/transformer/` directory exists

### Vision encoder error (I2V mode)
- Request access to FLUX.1-Redux-dev on HuggingFace
- Set `HF_TOKEN` and re-run `./download_models.sh`

## References

- [GitHub Repository](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
- [HuggingFace Models](https://huggingface.co/tencent/HunyuanVideo-1.5)
- [Technical Report](https://arxiv.org/abs/2511.18870)

## License

Tencent Hunyuan Community License. See [LICENSE](LICENSE) for details.
