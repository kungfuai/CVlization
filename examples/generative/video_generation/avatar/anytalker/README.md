# AnyTalker - Audio-Driven Talking Video Generation

AnyTalker generates talking head videos from a single image and audio input. Built on Alibaba's Wan2.1 model, it creates lip-synced videos with natural head movements and expressions.

## Features

- Single-person talking video generation
- Audio-driven lip sync
- Natural head movements and expressions
- 480p output at 24 FPS

## Requirements

- NVIDIA GPU with 24GB+ VRAM (A10, RTX 4090, A100)
- Docker with NVIDIA runtime

## Usage

### Build

```bash
./build.sh
```

### Run inference

```bash
./predict.sh --image examples/images/1p-0.png --audio examples/audios/1p-0.wav --output outputs/result.mp4
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--image` | Input image path | Required |
| `--audio` | Input audio path (WAV) | Required |
| `--output` | Output video path | `outputs/output.mp4` |
| `--caption` | Scene description | "A person is talking." |
| `--skip-download` | Skip model download check | False |

## Model Weights

Models are automatically downloaded on first run to `~/.cache/cvlization/anytalker/checkpoints/`:

- `Wan2.1-Fun-V1.1-1.3B-InP` (~3GB) - Base diffusion model
- `AnyTalker-1.3B` (~2GB) - Fine-tuned weights
- `wav2vec2-base-960h` (~400MB) - Audio encoder

## Notes

- First run downloads ~5GB of model weights
- Generation takes 30-60 seconds per video on A10 GPU
- Uses PyTorch native SDPA (no flash_attn required)

## References

- [AnyTalker GitHub](https://github.com/HKUST-C4G/AnyTalker)
- [Paper](https://arxiv.org/abs/2505.00000) (placeholder)
