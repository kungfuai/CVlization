# LTX-2

Audio-video generation using Lightricks' LTX-2 model.

## Overview

LTX-2 is a 19B parameter DiT-based model that generates video with synchronized audio from text prompts or images. It uses an asymmetric dual-stream architecture (14B video + 5B audio streams) with bidirectional cross-modal attention.

## Requirements

- **GPU**: NVIDIA GPU with 48GB+ VRAM
  - FP8 mode: ~35GB peak VRAM
  - Without FP8: ~48GB peak VRAM
- **Disk**: ~50GB for model weights (FP8 mode, downloaded on first run)
- **Docker**: With NVIDIA runtime

## Usage

### Build

```bash
cvl run ltx2 build
```

### Run Inference

```bash
# Text-to-video (distilled pipeline - fast)
cvl run ltx2 predict -- --prompt "A serene mountain landscape at sunset"

# Text-to-video (full pipeline - higher quality)
cvl run ltx2 predict -- --pipeline full --prompt "A cat playing with yarn"

# Image-to-video
cvl run ltx2 predict -- --image portrait.jpg --prompt "The person smiles and waves"

# Quick test (smaller resolution, fewer frames)
cvl run ltx2 predict -- --height 512 --width 768 --num-frames 33

# Disable FP8 for higher quality (requires more VRAM)
cvl run ltx2 predict -- --no-fp8

# Audio-conditioned generation (injects audio latents)
cvl run ltx2 predict -- --audio speech.wav --audio-strength 0.8 --prompt "A presenter speaks in sync"
```

### LoRA Usage

```bash
# List available LoRA files in a HuggingFace repo
cvl run ltx2 predict -- --list-lora-files kabachuha/ltx2-inflate-it

# Use a LoRA by repo ID (auto-selects a .safetensors file if only one exists)
cvl run ltx2 predict -- --prompt "..." --lora kabachuha/ltx2-inflate-it 1.0

# Use a specific file in a repo
cvl run ltx2 predict -- --prompt "..." --lora kabachuha/ltx2-inflate-it::ltx2-inflate.safetensors 1.0

# Use a local LoRA file
cvl run ltx2 predict -- --prompt "..." --lora /path/to/lora.safetensors 0.8
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pipeline` | distilled | Pipeline: `distilled` (fast) or `full` (quality). `two_stage` is deprecated. |
| `--prompt` | sample | Text prompt describing the video |
| `--image` | - | Input image for image-to-video |
| `--audio` | - | Input audio for audio-latent conditioning |
| `--audio-strength` | 1.0 | Audio conditioning strength (0.0 to 1.0) |
| `--output` | output.mp4 | Output video path |
| `--height` | 1024 | Video height (divisible by 64) |
| `--width` | 1536 | Video width (divisible by 64) |
| `--num-frames` | 121 | Number of frames (~5 seconds at 24 FPS) |
| `--frame-rate` | 24.0 | Frame rate (FPS) |
| `--seed` | 10 | Random seed |
| `--num-inference-steps` | 40 | Denoising steps (full pipeline only) |
| `--cfg-guidance-scale` | 4.0 | CFG scale (full pipeline only) |
| `--lora` | - | LoRA model path and optional strength (can be repeated) |
| `--list-lora-files` | - | List .safetensors files in a HuggingFace LoRA repo and exit |
| `--no-fp8` | false | Disable FP8 mode |
| `--enhance-prompt` | false | Use model's prompt enhancement |

## Pipelines

| Pipeline | Steps | CFG | Speed | Quality | VRAM (FP8) | VRAM (Full) |
|----------|-------|-----|-------|---------|------------|-------------|
| distilled | 8+4 | No | Fast (~16s) | Good | ~35GB | ~48GB |
| full | 40+4 | Yes | Slow | Best | ~40GB* | ~52GB* |

*Estimated; full pipeline uses more memory for CFG.

## Output

- **Resolution**: Up to 1536x1024 (two-stage upsampling)
- **Frame rate**: 24 FPS default
- **Audio**: 24kHz stereo, synchronized with video
- **Format**: MP4 with H.264 video + AAC audio

## Model

- **Parameters**: 19B (14B video + 5B audio)
- **Architecture**: Asymmetric dual-stream DiT with bidirectional cross-attention
- **Text encoder**: Gemma 3-12B
- **Video VAE**: 32x spatial, 8x temporal compression
- **Audio VAE**: 4x temporal, stereo support

### Model Files (~50GB for FP8)

Downloaded automatically on first run:
- Main checkpoint: 26GB (FP8) or ~38GB (full precision)
- Spatial upsampler: ~1GB
- Distilled LoRA: ~1GB (full pipeline only)
- Gemma text encoder: 23GB

## Limitations

- FP8 mode requires ~35GB VRAM (works on 48GB GPUs, tight fit on 40GB)
- Full precision requires ~48GB VRAM
- First run downloads ~50GB of model weights (FP8 mode)
- Consumer GPUs (RTX 4090 24GB, etc.) do not have enough VRAM
- Generation time: ~16s for 33 frames at 512x768
- Audio quality varies with prompt specificity

## Links

- [GitHub](https://github.com/Lightricks/LTX-2)
- [HuggingFace](https://huggingface.co/Lightricks/LTX-2)
- [Technical Report](https://videos.ltx.io/LTX-2/grants/LTX_2_Technical_Report_compressed.pdf)
