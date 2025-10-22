# Wan2GP

This is adapted from [Wan2GP](https://github.com/deepbeepmeep/Wan2GP).

Wan2GP provides state-of-the-art text-to-video (T2V) and image-to-video (I2V) generation using diffusion models. This implementation supports both CLI-based generation and an interactive Gradio web interface.

## Prerequisites

- Docker with GPU support
- NVIDIA GPU with at least 24GB VRAM (for 14B models)
- Downloaded model checkpoints (see below)

## Setup

### 1. Download Models

First, download the pre-trained models:

```bash
bash examples/video_gen/wan2gp/download_models.sh
```

This will download the model checkpoints to a local directory. The models are quite large (several GB each).

### 2. Build Docker Image

Build the Docker image with all dependencies:

```bash
bash examples/video_gen/wan2gp/build.sh
```

## Usage

### Command-Line Interface (Recommended for Automation)

The `predict.py` script provides a CLI for batch generation and scripting.

> **Note**: The CLI script is newly added and not fully tested. It should work after building the Docker image, but you may need to adjust paths or parameters. The Gradio web interface below is the tested and recommended way to use this example.

#### Text-to-Video Generation

Generate a video from a text prompt:

```bash
# Navigate to the wan2gp directory
cd examples/generative/video_generation/wan2gp

# Using 1.3B model (faster, lower VRAM)
bash predict.sh t2v \
  --prompt "A cat playing piano in a cozy room" \
  --checkpoint-dir /workspace/models \
  --output cat_piano.mp4

# Using 14B model (higher quality, more VRAM)
bash predict.sh t2v \
  --prompt "A sunset over the ocean with waves crashing" \
  --model-size 14B \
  --checkpoint-dir /workspace/models \
  --output sunset.mp4 \
  --steps 50 \
  --guidance-scale 7.5
```

#### Image-to-Video Generation

Animate an existing image:

```bash
# From the wan2gp directory
cd examples/generative/video_generation/wan2gp

bash predict.sh i2v \
  --image /workspace/input.jpg \
  --prompt "The scene comes to life with gentle movement" \
  --checkpoint-dir /workspace/models \
  --output animated.mp4 \
  --resolution 720p
```

#### Advanced Options

```bash
# From the wan2gp directory
cd examples/generative/video_generation/wan2gp

# Generate with specific parameters
bash predict.sh t2v \
  --prompt "A futuristic city at night" \
  --checkpoint-dir /workspace/models \
  --output city.mp4 \
  --frames 81 \           # Number of frames (must be 4n+1)
  --fps 8 \               # Output framerate
  --size 1280x720 \       # Resolution
  --steps 50 \            # Sampling steps (higher = better quality)
  --guidance-scale 5.0 \  # CFG scale (higher = more prompt adherence)
  --shift 5.0 \           # Noise schedule shift (3.0 for 480p, 5.0 for 720p+)
  --seed 42               # Reproducible results

# View all options
bash predict.sh t2v --help
bash predict.sh i2v --help
```

### Interactive Web Interface

For interactive experimentation, start the Gradio web UI:

```bash
bash examples/video_gen/wan2gp/up.sh
```

The app will be available at [http://localhost:7860](http://localhost:7860).

The web interface provides:
- Real-time parameter adjustment
- Multiple model selection (1.3B, 14B)
- Prompt expansion with AI assistance
- Preview and comparison tools

## Model Specifications

| Model | Parameters | VRAM Required | Speed | Quality |
|-------|------------|---------------|-------|---------|
| T2V 1.3B | 1.3 billion | ~12-16 GB | Fast | Good |
| T2V 14B | 14 billion | ~24-32 GB | Slower | Excellent |
| I2V 14B | 14 billion | ~24-32 GB | Slower | Excellent |

## Tips for Best Results

1. **Frame Count**: Must be `4n+1` (e.g., 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81)
2. **Shift Parameter**: Use `3.0` for 480p videos, `5.0` for 720p+ videos
3. **Sampling Steps**: 40-50 steps provide good balance; 60+ for highest quality
4. **Guidance Scale**: 5.0-7.5 works well; higher values increase prompt adherence but may reduce diversity
5. **VRAM Optimization**: Use `--no-offload` flag if you have sufficient VRAM for faster generation

## Troubleshooting

**Out of Memory (OOM) errors:**
- Use the 1.3B model instead of 14B
- Reduce frame count (e.g., 41 instead of 81)
- Reduce resolution (480p instead of 720p)
- Enable model offloading (default behavior)

**Slow generation:**
- Model offloading to CPU saves VRAM but is slower
- Use `--no-offload` if you have sufficient VRAM
- Reduce sampling steps (30-40 instead of 50+)

## Files

- `predict.py` - CLI script for batch generation
- `predict.sh` - Docker wrapper for CLI script
- `gradio_server.py` - Web interface server
- `up.sh` - Start web interface
- `build.sh` - Build Docker image
- `download_models.sh` - Download model checkpoints

## Citation

Based on the Wan video generation model from Alibaba. For more information, see the [upstream repository](https://github.com/deepbeepmeep/Wan2GP).






