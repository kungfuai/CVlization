# LongCat-Video

Foundational video generation using Meituan's LongCat-Video model (13.6B parameters).

## Overview

LongCat-Video supports multiple generation modes:
- **T2V (Text-to-Video)**: Generate video from text prompt only
- **I2V (Image-to-Video)**: Generate video from reference image + text prompt
- **Long Video**: Multi-segment video generation for longer content

The model uses Flow Matching with a 13.6B parameter DiT backbone.

## Requirements

- **GPU**: NVIDIA GPU with 48GB+ VRAM recommended
- **Disk**: ~60GB for model weights
- **Docker**: With NVIDIA runtime

## Usage

### Build

```bash
cvl run longcat_video build
# or
./build.sh
```

### Run Inference

```bash
# Image-to-Video (default mode) with sample image
cvl run longcat_video predict

# Text-to-Video
cvl run longcat_video predict -- --mode t2v --prompt "A cat playing with a ball"

# Image-to-Video with custom image
cvl run longcat_video predict -- --mode i2v --image photo.jpg --prompt "A person smiling"

# Long Video (multiple segments)
cvl run longcat_video predict -- --mode long_video --segments 5 --prompt "A scenic mountain landscape"
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | i2v | Generation mode: t2v, i2v, long_video |
| `--image` | sample | Input image path (for i2v mode) |
| `--prompt` | (default) | Text prompt describing the video |
| `--negative-prompt` | (default) | Negative prompt |
| `--output` | outputs/output.mp4 | Output video path |
| `--resolution` | 480p | Output resolution: 480p or 720p |
| `--frames` | 93 | Frames per segment |
| `--steps` | 50 | Inference steps |
| `--guidance` | 4.0 | Guidance scale |
| `--segments` | 3 | Number of segments (long_video mode) |
| `--seed` | 42 | Random seed |

## Generation Modes

### Text-to-Video (t2v)
Generates video purely from text description. No input image required.

```bash
cvl run longcat_video predict -- --mode t2v \
    --prompt "A white cat sits on a sunny windowsill, watching birds fly by" \
    --steps 50
```

### Image-to-Video (i2v)
Animates a reference image based on the text prompt.

```bash
cvl run longcat_video predict -- --mode i2v \
    --image portrait.jpg \
    --prompt "A person smiles and turns their head slowly"
```

### Long Video (long_video)
Generates extended videos by chaining multiple segments. Uses video continuation
to maintain temporal consistency.

```bash
cvl run longcat_video predict -- --mode long_video \
    --segments 5 \
    --prompt "A skateboarder rides along a winding mountain road"
```

## Performance

Tested on NVIDIA RTX PRO 6000 Blackwell (96GB VRAM):

| Metric | Value |
|--------|-------|
| **Model Download** | ~30GB |
| **Model Loading** | ~30 seconds |
| **I2V (10 steps, 33 frames)** | ~55 seconds |
| **Speed** | ~5.5 seconds per denoising step |

Notes:
- First run downloads the model (~30GB) which takes ~11 minutes
- VRAM usage varies by resolution and frame count
- Use `--steps 10` for faster testing (lower quality)

## Output Specifications

| Resolution | Dimensions | FPS | Frames/Segment |
|------------|-----------|-----|----------------|
| 480p | 832 x 480 | 15 | 93 (~6.2s) |
| 720p | 1280 x 768 | 15 | 93 (~6.2s) |

For long_video mode, each additional segment adds ~80 new frames (minus overlap).

## Model Details

- **Architecture**: 13.6B parameter DiT with Flow Matching
- **Components**: UMT5 text encoder, WAN VAE, Flow Match scheduler
- **Output**: 480P or 720P video at 15 FPS
- **Source**: [meituan-longcat/LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video)

## Links

- [LongCat-Video GitHub](https://github.com/meituan-longcat/LongCat-Video)
- [HuggingFace Model](https://huggingface.co/meituan-longcat/LongCat-Video)
