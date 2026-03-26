# V3: Continuous Space-Time Video Super-Resolution

V3 uses 3D Fourier Fields for continuous space-time video super-resolution, supporting arbitrary spatial and temporal upsampling factors. Based on a JAX/Flax implementation with deformable attention (ICLR 2026).

## Sample Output

Side-by-side: bicubic 2x upscale (left) vs V3 super-resolved (right) on Vid4 "city":

![comparison](https://huggingface.co/datasets/zzsi/cvl/resolve/main/v3_vsr/comparison.gif)

**Metrics** (Vid4 city, 2x spatial + 2x temporal):

| PSNR (dB) | SSIM | tOF |
|-----------|------|-----|
| 35.39 | 0.960 | 0.110 |

## What to Expect

- **First run**: downloads ~54MB checkpoint from Google Drive + ~22MB Vid4 sample from HuggingFace, cached to `~/.cache/cvlization/v3/`
- **What it does**: upsamples video frames spatially (e.g., 4x) and temporally (e.g., 2x-8x)
- **Default sample**: Vid4 "city" clip (34 frames, 704x576) with 2x spatial + 2x temporal upsampling
- **Output**: PNG frames in the output directory, optionally a side-by-side comparison MP4
- **Runtime**: ~3 min for the default sample on Blackwell GPU

## Requirements

- NVIDIA GPU with >= 24GB VRAM (RTX 3090/4090/A100/H100)
- Docker with NVIDIA runtime

## Build

```bash
bash examples/perception/video_super_resolution/v3/build.sh
```

## Run

Default (Vid4 city sample, 2x spatial + 2x temporal):

```bash
cvl run v3-vsr predict
```

With comparison video and evaluation:

```bash
bash examples/perception/video_super_resolution/v3/predict.sh \
  --space-scale 2 --time-scale 2 \
  --output-video comparison.mp4 --eval
```

Custom video (directory of PNG frames):

```bash
bash examples/perception/video_super_resolution/v3/predict.sh \
  --input /path/to/frames_dir \
  --space-scale 4 --time-scale 8 \
  --output-video result.mp4
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `vid4_city` | Input video directory or built-in sample name |
| `--output` | `v3_result` | Output directory for super-resolved frames |
| `--space-scale` | `4` | Spatial upsampling factor |
| `--time-scale` | `2` | Temporal upsampling factor |
| `--output-video` | (none) | Side-by-side comparison MP4 path |
| `--fps` | `12` | FPS for output video |
| `--eval` | off | Run PSNR/SSIM evaluation against GT |
| `--checkpoint` | (auto-download) | Path to V3 checkpoint file |
| `--cache-dir` | `~/.cache/cvlization/v3` | Cache directory for checkpoint |
| `--verbose` | off | Enable verbose logging |

## References

- Repository: https://github.com/prs-eth/v3
- Paper: https://arxiv.org/abs/2509.26325
- Checkpoint: https://drive.google.com/file/d/15nw5NhEIf7VvetEtQI1cnrLNWPi_9FGj
