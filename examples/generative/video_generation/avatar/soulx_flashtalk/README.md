# SoulX-FlashTalk

Audio-driven streaming talking-avatar video generation. Given a reference image
and an audio clip, SoulX-FlashTalk generates a lip-synced talking-head video.

The model is built on InfiniteTalk / Wan2.1, distilled to **4 sampling steps**
via DMD2 + Self-Forcing++ ("Self-Correcting Bidirectional Distillation"). It
generates video **chunk-by-chunk** (autoregressive, with motion-frame
conditioning carried between chunks), so it supports arbitrarily long clips.

## Features

- **Audio-to-Video Lip Sync**: talking-head video synchronized to input audio
- **Streaming / infinite length**: chunked autoregressive generation
- **Runs on a 40GB GPU**: uses int8 `optimum-quanto` quantization plus
  `--cpu_offload` (no FP8, so it works on Ampere/A100, not just Hopper)

## Requirements

### Hardware

- **GPU**: 1x NVIDIA GPU, ≥40GB VRAM with CPU offload (≥64GB without)
- **RAM**: 64GB+ system memory
- **Disk**: ~55GB for models

CPU offload is enabled by default and is required to fit a 40-48GB card. It is
not real-time on a single GPU (~35s per 33-frame chunk on an A100-40GB);
real-time generation requires a multi-GPU H800-class setup (see upstream repo).

## Quick Start

```bash
# 1. Build the Docker image
cvl run soulx_flashtalk build

# 2. Generate a video (models download automatically on first run)
cvl run soulx_flashtalk predict --output output.mp4
```

With no `--audio` / `--image`, the bundled upstream sample
(`man.png` + `cantonese_16k.wav`) is used. To use your own inputs:

```bash
cvl run soulx_flashtalk predict \
    --audio input.wav \
    --image reference.png \
    --output output.mp4
```

## Usage

### Build

```bash
cvl run soulx_flashtalk build
```

### Predict

Models are downloaded automatically on first run:

- `Soul-AILab/SoulX-FlashTalk-14B`: DiT, VAE, T5 weights (~50GB)
- `TencentGameMate/chinese-wav2vec2-base`: audio encoder (~1.5GB)

```bash
cvl run soulx_flashtalk predict --audio input.wav --image reference.png --output output.mp4
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--audio` | bundled sample | Input audio file (WAV) |
| `--image` | bundled sample | Reference image (PNG, JPG) |
| `--output` | output.mp4 | Output video path |
| `--prompt` | "A person is talking..." | Text prompt describing the scene |
| `--seed` | 9999 | Random seed |
| `--audio-encode-mode` | stream | `stream` (per-chunk) or `once` (all audio) |
| `--no-cpu-offload` | (off) | Disable CPU offload (needs >64GB VRAM) |
| `--verbose` | (off) | Enable verbose framework logging |

## Model Downloads & Caching

- Weights are fetched lazily at runtime and cached under `~/.cache/huggingface/`.
- `predict.sh` mounts the cache into the container so subsequent runs reuse weights.

## How it works

`predict.py` is a thin wrapper around the upstream `generate_video.py`:

1. Resolves input/output paths (CVL workspace mount aware).
2. Lazily downloads the two model repos from HuggingFace.
3. Runs the upstream generator (`cwd` set to the cloned repo, since it loads
   `flash_talk/configs/infer_params.yaml` via a relative path).
4. The upstream `save_video()` writes a silent video, then ffmpeg-muxes the
   input audio into it. The wrapper passes a `res_`-prefixed `--save_file` so
   upstream's temp-path logic (`video_path.replace('res_', '')`) yields a
   distinct temp file, then moves the result to `--output`.

## Verification

Verified on an NVIDIA A100-PCIE-40GB host (`acasia`):

- **Docker build**: the image builds successfully (~30GB). The Python 3.11
  base provides torch 2.7.1+cu128; the cp311 flash-attn wheel and all frozen
  requirements install cleanly.
- **End-to-end inference**: ran `predict.py` inside the container (single GPU,
  `--cpu_offload`, bundled sample). Weights loaded from the HuggingFace cache,
  all 34 chunks generated (~35s per 33-frame chunk; GPU at 100% util,
  **39.2GB VRAM** — fits a 40GB card), audio ffmpeg-muxed inside the
  container, and the output written to the mounted path: a 37.5s, 448x768,
  25fps MP4 with audio.

Not verified: the `cvl run` CLI path specifically — the example was exercised
via a direct `docker run` mirroring `predict.sh`'s mounts.

## Notes

- This model is the A100-friendly alternative to the sibling `live_avatar`
  example: LiveAvatar's low-VRAM path needs FP8, which requires Hopper-class
  GPUs. SoulX-FlashTalk uses int8 quantization and runs on Ampere.

## References

- Paper: https://arxiv.org/abs/2512.23379
- GitHub: https://github.com/Soul-AILab/SoulX-FlashTalk
- Project Page: https://soul-ailab.github.io/soulx-flashtalk/
- Base models: [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk), [Wan2.1](https://github.com/Wan-Video/Wan2.1)

## License

See the upstream repository for license terms.
