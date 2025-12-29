# LiveAvatar

Generate streaming real-time audio-driven avatar video using LiveAvatar from Alibaba.

## Features

- **Audio-to-Video Lip Sync**: Generate talking head video synchronized to audio
- **LoRA Fine-tuning**: Uses lightweight LoRA adapter on Wan2.2-S2V-14B base model
- **Single GPU Support**: Runs on a single 80GB+ GPU with model offloading
- **Infinite Length**: Supports streaming generation for long-form content

## Requirements

### Hardware
- **GPU**: 1x 80GB+ VRAM (H100, H200, A100-80GB)
- **RAM**: 64GB+ system memory
- **Disk**: ~60GB for models

## Quick Start

```bash
# 1. Build the Docker image
cvl run live_avatar build

# 2. Generate video from audio (models download automatically on first run)
cvl run live_avatar predict --audio input.wav --image avatar.jpg --output output.mp4
```

### Sample Inputs

You can use sample files from other CVlization examples:

```bash
# Audio sample
../lite_avatar/funasr_local/runtime/triton_gpu/client/test_wavs/mid.wav

# Portrait sample
../fastavatar/data/example.png
```

Or use LiveAvatar's bundled examples (after build):
```bash
# Examples are in /workspace/LiveAvatar/examples/ inside container
cvl run live_avatar predict \
    --audio /workspace/LiveAvatar/examples/dwarven_blacksmith.wav \
    --image /workspace/LiveAvatar/examples/dwarven_blacksmith.jpg \
    --output output.mp4
```

## Usage

### Build

```bash
cvl run live_avatar build
```

### Predict

Models are downloaded automatically on first run:
- `Wan-AI/Wan2.2-S2V-14B`: Base video generation model (~50GB)
- `Quark-Vision/Live-Avatar`: LoRA fine-tuned weights (~500MB)

```bash
cvl run live_avatar predict \
    --audio input.wav \
    --image reference.jpg \
    --output output.mp4
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--audio` | required | Input audio file (WAV) |
| `--image` | required | Reference image (PNG, JPG) |
| `--output` | output.mp4 | Output video path |
| `--prompt` | "A person talking..." | Text prompt describing scene |
| `--size` | 704*384 | Output video size (width*height) |
| `--sample_steps` | 4 | Number of denoising steps |
| `--infer_frames` | 48 | Frames per clip |
| `--num_clips` | 100 | Max number of clips to generate |
| `--seed` | 42 | Random seed |
| `--offload` | True | Offload model to CPU when not in use |

## Performance

Single GPU (80GB+ VRAM) with offloading:
- ~5-10 seconds per video block
- Quality may degrade for very long videos without online decode

**Measured Metrics (RTX PRO 6000 Blackwell 96GB, 4 steps, 704x384, 48 frames/clip):**
- Latency: <0.01s (model ready to generation start)
- Throughput: ~0.8 fps

For real-time 20 FPS generation, use 5x H800 GPUs (see upstream repo).

## Architecture

LiveAvatar uses a block-wise autoregressive approach:

```
┌─────────────────────────────────────────────────────────────┐
│                    Single GPU Pipeline                       │
│  - T5 Text Encoder (offloaded)                              │
│  - Audio Encoder                                             │
│  - VAE Encoder/Decoder (offloaded)                          │
│  - DiT with LoRA (Wan2.2-S2V-14B + LiveAvatar adapter)      │
└─────────────────────────────────────────────────────────────┘
```

## Model Details

### LiveAvatar
- LoRA fine-tuning on Wan2.2-S2V-14B
- 4-step denoising with distribution-matching distillation
- Paper: [arxiv.org/abs/2512.04677](https://arxiv.org/abs/2512.04677)

### Wan2.2-S2V-14B
- 14B parameter Speech-to-Video model
- Causal autoregressive architecture
- Block-wise generation with KV-caching

## References

- [LiveAvatar GitHub](https://github.com/Alibaba-Quark/LiveAvatar)
- [LiveAvatar Paper](https://arxiv.org/abs/2512.04677)
- [Project Page](https://liveavatar.github.io/)

## License

Apache 2.0 (LiveAvatar and Wan model)
