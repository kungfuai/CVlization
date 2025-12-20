# IMTalker

Audio-driven talking face generation using implicit motion transfer with flow matching.

## Overview

IMTalker generates high-quality talking face videos from a single image and audio file. It uses:
- **Wav2Vec2** for audio feature extraction
- **Flow Matching** with implicit motion transfer for motion generation
- **Neural renderer** for final frame synthesis

Key features:
- 40+ FPS inference on RTX 4090
- 512x512 output resolution
- Supports both audio-driven and video-driven generation
- Apache 2.0 license

## Requirements

- NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- Docker with NVIDIA Container Toolkit
- ~2.3GB disk space for model checkpoints

### GPU Compatibility

This example uses PyTorch 2.9.0 with CUDA 12.8, supporting a wide range of GPUs:
- GTX 900/1000 series, RTX 20/30/40/50 series
- Tesla V100, A100, H100, Blackwell GPUs

## Usage

### Build the Docker image

```bash
./build.sh
```

### Generate a talking video

```bash
./predict.sh --image examples/images/face.png --audio examples/audios/speech.wav --output outputs/result.mp4
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--image` | Path to input face image | Required |
| `--audio` | Path to input audio (WAV) | Required |
| `--output` | Output video path | `outputs/output.mp4` |
| `--no-crop` | Disable automatic face cropping | False |
| `--cfg-scale` | Audio classifier-free guidance scale | 3.0 |
| `--nfe` | Number of function evaluations for ODE solver | 10 |
| `--seed` | Random seed | 42 |

## Model Downloads

Models are automatically downloaded from HuggingFace on first run:
- `cbsjtu01/IMTalker` - Generator and renderer checkpoints (~2GB)
- `facebook/wav2vec2-base-960h` - Audio encoder (~360MB)

Checkpoints are cached in `~/.cache/cvlization/imtalker/checkpoints/`.

## Input Requirements

- **Image**: JPG, PNG (any resolution, auto-cropped to 512x512)
- **Audio**: WAV format, 16kHz recommended (auto-resampled if needed)

## Output

- MP4 video at 512x512 resolution, 25 FPS
- H.264 video codec with AAC audio

## References

- [IMTalker Paper](https://arxiv.org/abs/2312.09614)
- [IMTalker GitHub](https://github.com/bigai-nlco/IMTalker)
- [HuggingFace Model](https://huggingface.co/cbsjtu01/IMTalker)
