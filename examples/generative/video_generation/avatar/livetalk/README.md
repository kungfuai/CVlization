# LiveTalk - Real-Time Avatar Video Generation

Real-time multimodal avatar video generation from a reference image and audio.
Generates talking avatar videos with lip-sync at 24.82 FPS using 4-step diffusion.

**Paper:** [arXiv:2512.23576](https://arxiv.org/abs/2512.23576)
**Repository:** https://github.com/GAIR-NLP/LiveTalk
**License:** CC-BY-NC-SA 4.0 (Non-commercial use only)

## Requirements

- **GPU:** NVIDIA GPU with 24GB+ VRAM (RTX 4090, A10, A100, etc.)
- **RAM:** 64GB system memory
- **Docker:** With NVIDIA Container Toolkit

## Quick Start

```bash
# Build the Docker image
./build.sh

# Run inference with default example
./predict.sh

# Run with custom inputs
./predict.sh --image /path/to/face.jpg --audio /path/to/speech.wav --output result.mp4
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image` | example.jpg | Reference face image (PNG/JPG) |
| `--audio` | example.wav | Speech audio (WAV, 16kHz) |
| `--output` | outputs/output.mp4 | Output video path |
| `--duration` | 5 | Video duration in seconds (must be 3n+2: 5, 8, 11, 14...) |
| `--max-hw` | 720 | Max resolution (720=480p, 1280=720p) |
| `--fps` | 16 | Output video FPS |
| `--prompt` | (default) | Text prompt describing the video |
| `--skip-download` | false | Skip model weight download |

## Model Weights

Models are automatically downloaded on first run (~10GB total):

- **Wan2.1-T2V-1.3B** - Base video diffusion model
- **LiveTalk-1.3B-V0.1** - Distilled 4-step avatar model
- **wav2vec2-base-960h** - Audio feature extraction

Weights are cached in `~/.cache/cvlization/livetalk/weights/`.

## Technical Details

- **Architecture:** Causal Video Diffusion Transformer (DiT) with KV cache
- **Inference:** 4 denoising steps (distilled from ~83s baseline)
- **Performance:** 24.82 FPS, 0.33s first-frame latency
- **Audio sync:** Wav2Vec2 audio encoding with block-aligned features
- **Identity preservation:** Anchor-Heavy Identity Sinks (AHIS)

## Notes

- Video duration must follow the formula `3n + 2` (e.g., 5, 8, 11, 14, 17, 20 seconds)
- Input audio should be 16kHz WAV format
- Reference image should show a clear frontal face
- The model uses ~20GB GPU memory during inference

## Citation

```bibtex
@article{livetalk2024,
  title={Real-Time Multimodal Interactive Video Diffusion via Improved On-Policy Distillation},
  author={Chern, Ethan and Hu, Zhulin and Tang, Bohao and Su, Jiadi and Chern, Steffi and Deng, Zhijie and Liu, Pengfei},
  journal={arXiv preprint arXiv:2512.23576},
  year={2024}
}
```
