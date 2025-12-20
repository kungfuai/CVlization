# PersonaLive

Real-time expressive portrait animation for live streaming. Takes a reference portrait image and a driving video to generate animated portraits that follow the motion.

## Features

- **Portrait Animation**: Animate any portrait using motion from a driving video
- **Real-time Capable**: Designed for streaming applications on 12GB GPUs
- **Diffusion-based**: Uses Stable Diffusion backbone with custom motion encoding
- **Temporal Consistency**: Maintains consistency across frames with temporal windowing

## Requirements

- NVIDIA GPU with 12GB+ VRAM (tested on RTX 3080/4080/4090)
- CUDA 12.4+
- ~15GB disk space for model weights

## Setup

### 1. Build the Docker image

```bash
./build.sh
```

### 2. (Optional) Pre-download model weights

Models are downloaded lazily on first run and cached in `~/.cache/huggingface`.
To pre-populate the cache:

```bash
./download_models.sh
```

## Model Downloads & Caching

- All models are fetched lazily at runtime via HuggingFace Hub
- Cached under `$HF_HOME` (defaults to `~/.cache/huggingface`)
- The Docker `predict.sh` mounts this cache so subsequent runs reuse weights
- Total download size: ~8GB (PersonaLive ~2GB, SD base model ~5GB, VAE ~350MB)

## Quick Start

```bash
# Run with demo assets (downloads models on first run)
./predict.sh

# Custom inputs
./predict.sh --ref_image my_portrait.jpg --driving_video motion.mp4

# Adjust quality/speed tradeoff
./predict.sh --steps 8 --max_frames 200
```

## Usage

```bash
./predict.sh [OPTIONS]

Required inputs (defaults to demo assets):
  --ref_image PATH        Reference portrait image to animate
  --driving_video PATH    Video providing motion/expressions

Optional:
  --output, -o PATH       Output video path (default: outputs/output_TIMESTAMP.mp4)
  --width, -W INT         Output width (default: 512)
  --height, -H INT        Output height (default: 512)
  --max_frames, -L INT    Max frames to process (default: 100)
  --steps INT             Diffusion steps (default: 4, higher = better quality)
  --seed INT              Random seed (default: 42)
  --verbose               Enable verbose logging
```

## Examples

```bash
# Basic usage with defaults
./predict.sh

# Higher quality (more steps)
./predict.sh --steps 8

# Process more frames
./predict.sh --max_frames 200 --output outputs/long_video.mp4

# Different resolution
./predict.sh --width 768 --height 768
```

## Performance

On RTX 4090 (24GB):
- 100 frames at 512x512: ~2-3 minutes
- 4 inference steps (default): fast, good quality
- 8 inference steps: slower, better quality

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--max_frames` to process fewer frames at once
- Reduce resolution with `--width 384 --height 384`

### Model not found
- Run `./download_models.sh` to download all required weights
- Check that `pretrained_weights/personalive/` contains the checkpoint files

### xformers error
- The Docker image includes xformers 0.0.29 for PyTorch 2.5.1
- If building locally, ensure xformers version matches your PyTorch version

## Architecture

PersonaLive uses a diffusion-based approach with:
- **Reference UNet (2D)**: Encodes the reference portrait
- **Denoising UNet (3D)**: Generates temporally consistent video
- **Motion Encoder**: Extracts motion features from driving video faces
- **Pose Guider**: Provides spatial guidance from 3D keypoints
- **LivePortrait Motion Extractor**: Extracts implicit 3D keypoints

## References

- Paper: https://arxiv.org/abs/2512.11253
- GitHub: https://github.com/GVCLab/PersonaLive
- HuggingFace: https://huggingface.co/huaichang/PersonaLive

## License

Note: The upstream repository does not include a license file. Use for research and evaluation purposes only.

## Acknowledgements

Built upon Moore-AnimateAnyone, X-NeMo, StreamDiffusion, RAIN, and LivePortrait.
