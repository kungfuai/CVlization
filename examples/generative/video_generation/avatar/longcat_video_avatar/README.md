# LongCat-Video-Avatar

Audio-driven talking head generation using Meituan's LongCat-Video-Avatar model.

## Overview

Generates videos of a person speaking, driven by audio input. The model animates lip movements and head motion based on the audio.

Supports two modes:
- **Single**: One person with one audio track
- **Multi**: Two people with separate audio tracks

## Requirements

- **GPU**: NVIDIA GPU with 64GB+ VRAM required (uses ~66GB during inference)
- **Disk**: ~40GB for model weights (base + avatar models)
- **Docker**: With NVIDIA runtime

## Performance

Tested on RTX PRO 6000 Blackwell (96GB VRAM):

| Metric | Value |
|--------|-------|
| Model loading | ~30 seconds |
| Per-step time | ~9.7 seconds |
| 5 steps, 33 frames | ~48 seconds denoising |
| 10 steps, 33 frames | ~96 seconds denoising |
| Peak VRAM | ~66 GB |
| Output (33 frames) | 2.2 seconds @ 15 FPS |

First run downloads ~40GB of model weights.

## Usage

### Build

```bash
cvl run longcat_video_avatar build
```

### Run Inference

```bash
# Single person with sample inputs (downloads sample image + audio)
cvl run longcat_video_avatar predict

# Faster test with fewer steps
cvl run longcat_video_avatar predict -- --steps 10 --frames 33

# Custom inputs
cvl run longcat_video_avatar predict -- --image portrait.jpg --audio speech.mp3

# Multi-person mode
cvl run longcat_video_avatar predict -- \
    --mode multi \
    --image two_people.jpg \
    --audio person1.mp3 \
    --audio2 person2.mp3
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | single | Generation mode: single or multi |
| `--image` | sample | Input image path |
| `--audio` | sample | Audio file for person 1 |
| `--audio2` | - | Audio file for person 2 (multi mode only) |
| `--audio-type` | para | Multi mode: para (parallel) or conv (conversation) |
| `--output` | output.mp4 | Output video path |
| `--resolution` | 480p | Output resolution: 480p or 720p |
| `--frames` | 93 | Number of frames (~6.2s at 15 FPS) |
| `--steps` | 50 | Inference steps (fewer = faster, lower quality) |
| `--text-guidance` | 4.0 | Text guidance scale |
| `--audio-guidance` | 4.0 | Audio guidance scale |
| `--seed` | 42 | Random seed |

## Output

| Resolution | Dimensions | FPS |
|------------|-----------|-----|
| 480p | 768 x 512 | 15 |
| 720p | 1280 x 768 | 15 |

Default 93 frames = ~6.2 seconds of video.

## Model

- **Parameters**: 13.6B (DiT transformer)
- **Architecture**: Flow Matching with DiT backbone
- **Audio encoder**: Wav2Vec2 (Chinese)
- **Models loaded**:
  - [meituan-longcat/LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video) (base: tokenizer, text encoder, VAE)
  - [meituan-longcat/LongCat-Video-Avatar](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar) (avatar transformer, audio encoder)

## Limitations

- Requires 64GB+ VRAM
- Chinese Wav2Vec2 audio encoder may work better with Chinese speech
- Multi-mode requires careful image composition (two people side by side)
- Long generation times (~10s per denoising step)

## Links

- [GitHub](https://github.com/meituan-longcat/LongCat-Video)
- [HuggingFace (Avatar)](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar)
- [HuggingFace (Base)](https://huggingface.co/meituan-longcat/LongCat-Video)
