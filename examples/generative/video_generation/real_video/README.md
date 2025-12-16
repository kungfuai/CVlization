# RealVideo

Generate lip-synced talking head video from audio input using Self-Forcing autoregressive diffusion.

## Features

- **Audio-to-Video Lip Sync**: Generate talking head video synchronized to audio
- **Custom Avatars**: Use any reference image as the talking head
- **High Quality**: Based on Wan2.2-S2V-14B (14B parameter model)
- **Fast Inference**: Self-Forcing with KV-caching for efficient generation

## Requirements

### Hardware
- **GPU**: 2x 80GB+ VRAM (H100, H200, A100-80GB, or Blackwell)
- **RAM**: 64GB+ system memory
- **Disk**: ~60GB for models

## Quick Start

```bash
# 1. Build the Docker image
cvl run real_video build

# 2. Generate video from audio (models download automatically on first run)
cvl run real_video predict --audio input.wav --image avatar.jpg --output output.mp4
```

### Sample Inputs

You can use sample files from other CVlization examples:

```bash
# Audio sample
../avatar/lite_avatar/funasr_local/runtime/triton_gpu/client/test_wavs/mid.wav

# Portrait sample
../avatar/fastavatar/data/example.png
```

Example with sample inputs:
```bash
cvl run real_video predict \
    --audio ../avatar/lite_avatar/funasr_local/runtime/triton_gpu/client/test_wavs/mid.wav \
    --image ../avatar/fastavatar/data/example.png \
    --output output.mp4
```

## Usage

### Build

```bash
cvl run real_video build
```

### Predict

Models are downloaded automatically on first run:
- `Wan-AI/Wan2.2-S2V-14B`: Base video generation model (~50GB)
- `Wan-AI/Wan2.1-T2V-1.3B`: VAE model (~5GB)
- `zai-org/RealVideo/model.pt`: Fine-tuned lip-sync checkpoint (~2GB)

```bash
cvl run real_video predict \
    --audio input.wav \
    --image reference.jpg \
    --output output.mp4
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--audio, -a` | required | Input audio file (WAV, MP3) |
| `--image, -i` | required | Reference image (PNG, JPG) |
| `--output, -o` | output.mp4 | Output video path |
| `--fps` | 16 | Output video framerate |
| `--width` | 480 | Output video width |
| `--height` | 640 | Output video height |
| `--denoising_steps` | 2 | Number of denoising steps |
| `--no_compile` | false | Disable torch.compile |

### Multi-GPU

Specify GPUs with `CUDA_VISIBLE_DEVICES`:

```bash
# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 cvl run real_video predict --audio input.wav --image ref.jpg

# Use GPUs 2,3,4,5 for faster generation
CUDA_VISIBLE_DEVICES=2,3,4,5 cvl run real_video predict --audio input.wav --image ref.jpg
```

## Performance

Generation time per video block (on H100):

| GPUs | 2 Denoising Steps | 4 Denoising Steps |
|------|-------------------|-------------------|
| 2    | ~385ms (compiled) | ~527ms (compiled) |
| 4    | ~306ms (compiled) | ~481ms (compiled) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               GPU 0 (VAE + Audio Encoder)                    │
│  - Reference image encoding                                  │
│  - Audio feature extraction                                  │
│  - Video frame decoding                                      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│               GPU 1+ (DiT Service)                           │
│  - Self-Forcing autoregressive diffusion                     │
│  - Wan2.2-S2V-14B with KV-caching                           │
│  - Sequence parallel inference                               │
└─────────────────────────────────────────────────────────────┘
```

## Model Details

### Self-Forcing
- Autoregressive video diffusion with KV-caching
- Bridges train-test gap for consistent streaming
- Paper: [arxiv.org/abs/2506.08009](https://arxiv.org/abs/2506.08009)

### Wan2.2-S2V-14B
- 14B parameter Speech-to-Video model
- Based on Wan2.1 architecture
- Optimized for audio-driven lip sync

## References

- [RealVideo GitHub](https://github.com/zai-org/RealVideo)
- [Self-Forcing Paper](https://arxiv.org/abs/2506.08009)
- [Wan2.1 Model](https://github.com/Wan-Video/Wan2.1)

## License

Apache 2.0 (RealVideo), see upstream repository for model licenses.
