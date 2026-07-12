# LingBot-Video

MoE video foundation model for embodied intelligence from Robbyant.

## Overview

LingBot-Video is the first large-scale open-source Mixture-of-Experts (MoE) video
foundation model, trained on 70,000+ hours of embodied and web video data. It features
physical-rationality and task-completion reward alignment, making it suited for
generating videos with physically plausible dynamics.

Two model variants are available:

| Variant | Parameters | Active Params | VRAM (bf16) | Download |
|---------|-----------|---------------|-------------|----------|
| Dense 1.3B | 1.3B | 1.3B | ~8GB | ~5GB |
| MoE 30B-A3B | 30B | ~3B | ~24GB+ | ~20GB |

## Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (dense) or 24GB+ VRAM (MoE)
- **Disk**: ~5GB for dense model, ~20GB for MoE model (downloaded on first run)
- **Docker**: With NVIDIA runtime

## Usage

### Build

```bash
cvl run lingbot_video build
```

### Run Inference

```bash
# Text-to-video with dense 1.3B model (default)
cvl run lingbot_video predict -- --prompt "A robotic arm picks up a red cube from a table"

# Text-to-video with MoE 30B model (needs 24GB+ VRAM)
cvl run lingbot_video predict -- --model moe-30b-a3b --prompt "A drone flies over a city at sunset"

# Text-to-image
cvl run lingbot_video predict -- --mode t2i --prompt "A robot in a factory" --output output.png

# Custom resolution and frame count
cvl run lingbot_video predict -- --height 480 --width 832 --num-frames 41 --steps 30

# Quick test (fewer frames, fewer steps)
cvl run lingbot_video predict -- --num-frames 21 --steps 20
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | dense-1.3b | Model variant: `dense-1.3b` or `moe-30b-a3b` |
| `--mode` | t2v | Generation mode: `t2v` (video) or `t2i` (image) |
| `--prompt` | sample | Text prompt describing the video to generate |
| `--negative-prompt` | - | Negative prompt for quality guidance |
| `--image` | - | Conditioning image for text+image-to-video (TI2V) |
| `--output` | output.mp4 | Output file path |
| `--height` | 480 | Video height (multiple of 16) |
| `--width` | 832 | Video width (multiple of 16) |
| `--num-frames` | 81 | Frame count (must be 4n+1, e.g. 21, 41, 61, 81) |
| `--fps` | 24.0 | Output video frame rate |
| `--steps` | 40 | Denoising steps |
| `--guidance-scale` | 3.0 | CFG guidance scale |
| `--seed` | 42 | Random seed |
| `--verbose` | false | Enable verbose logging |

## What to Expect

- **First run**: Downloads model weights (~5GB for dense, ~20GB for MoE) from HuggingFace, cached afterward
- **Output**: MP4 video file (or PNG for T2I mode) in the current directory
- **Runtime**: Dense 1.3B at 832x480, 21 frames, 10 steps: ~6s on RTX PRO 6000 Blackwell
- **Resolution**: Default 832x480 (landscape). Both dimensions must be multiples of 16
- **Duration**: Default 81 frames at 24 FPS = ~3.4 seconds of video

## Output

- **Video format**: MP4 (H.264)
- **Image format**: PNG (T2I mode)
- **Resolution**: Configurable, default 832x480
- **Frame rate**: Default 24 FPS

## Model Details

- **Architecture**: DiT (Diffusion Transformer) with MoE routing
- **Training data**: 70,000+ hours of embodied + web video
- **Reward alignment**: Physical rationality + task completion
- **License**: Apache 2.0
- **Paper**: [arXiv 2607.07675](https://huggingface.co/papers/2607.07675)

## Limitations

- The model uses `trust_remote_code=True` to load custom pipeline classes from HuggingFace
- The upstream project requires very recent dependencies (diffusers, transformers); compatibility may vary
- The prompt rewriter (for structured JSON captions) requires a separate 27B VLM and is not included; plain text prompts are used directly
- MoE 30B-A3B requires significantly more VRAM than the dense variant

## Links

- [GitHub](https://github.com/Robbyant/lingbot-video)
- [HuggingFace (Dense 1.3B)](https://huggingface.co/robbyant/lingbot-video-dense-1.3b)
- [HuggingFace (MoE 30B)](https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b)
- [Paper](https://huggingface.co/papers/2607.07675)
