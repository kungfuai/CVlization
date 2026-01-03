# Wan2.2-S2V-14B

Audio-driven talking head video generation using Alibaba's Wan2.2-S2V-14B model.

## Overview

Wan2.2-S2V-14B (Subject-to-Video) generates high-quality talking head videos from:
- A reference image (the subject)
- An audio file (speech or music)
- A text prompt (scene description)

The model features natural lip synchronization, head movements, and expressive gestures.

## Requirements

- **GPU**: NVIDIA GPU with 48GB+ VRAM (can use `--no-offload` for higher quality but requires more VRAM)
- **Disk**: ~60GB for model weights
- **Docker**: With NVIDIA runtime

## Usage

### Build

```bash
cvl run wan_s2v build
# or
./build.sh
```

### Run Inference

```bash
# With default sample inputs
cvl run wan_s2v predict

# With custom inputs
cvl run wan_s2v predict -- --image face.jpg --audio speech.wav --output result.mp4

# With custom prompt
cvl run wan_s2v predict -- --prompt "A cheerful person giving a presentation"
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | sample | Path to reference image (face) |
| `--audio` | sample | Path to audio file (WAV/MP3) |
| `--output` | outputs/output.mp4 | Output video path |
| `--prompt` | "A person is talking..." | Scene description |
| `--steps` | 40 | Sampling steps (fewer = faster, lower quality) |
| `--guidance` | 4.5 | Classifier-free guidance scale |
| `--frames` | 80 | Frames per clip (must be multiple of 4) |
| `--seed` | 42 | Random seed for reproducibility |
| `--no-offload` | False | Disable model offloading (requires more VRAM) |

## Performance

Tested on NVIDIA RTX PRO 6000 Blackwell (96GB VRAM):

| Metric | Value |
|--------|-------|
| **VRAM Usage** | ~40GB (with model offloading) |
| **Model Download** | ~30GB |
| **Speed** | ~65 seconds per diffusion step |
| **10-step inference** | ~12 min per 80-frame clip |
| **40-step inference** | ~45 min per 80-frame clip (estimated) |

Notes:
- Uses model offloading by default to reduce VRAM
- Longer audio generates multiple clips (each ~80 frames at 24 FPS = ~3.3s)
- Use `--steps 10` for faster testing (lower quality)

## Model Details

- **Architecture**: 14B parameter DiT with Flow Matching
- **Components**: T5 (umt5-xxl), VAE (Wan2.1), Wav2Vec2 for audio encoding
- **Output**: 480P or 720P video at 24 FPS
- **Source**: [Wan-AI/Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)

## Optional: TTS Mode

The model supports text-to-speech-to-video via CosyVoice TTS. This requires additional setup and is not enabled by default in this example.

## Links

- [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [Technical Report](https://humanaigc.github.io/wan-s2v-webpage/content/wan-s2v.pdf)
- [HuggingFace Model](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)
