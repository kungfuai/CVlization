# V3: Continuous Space-Time Video Super-Resolution

V3 uses 3D Fourier Fields for continuous space-time video super-resolution, supporting arbitrary spatial and temporal upsampling factors. Based on a JAX/Flax implementation with deformable attention (ICLR 2026).

## What to Expect

- **First run**: downloads ~54MB checkpoint from Google Drive + ~22MB Vid4 sample from HuggingFace, cached to `~/.cache/cvlization/v3/`
- **What it does**: upsamples video frames spatially (e.g., 4x) and temporally (e.g., 2x-8x)
- **Default sample**: Vid4 "city" clip (34 frames, 704x576) with 2x spatial + 2x temporal upsampling
- **Output**: PNG frames in the output directory
- **Runtime**: ~3 min for the default sample on Blackwell GPU

## Requirements

- NVIDIA GPU with >= 24GB VRAM (RTX 3090/4090/A100/H100)
- Docker with NVIDIA runtime

## Build

```bash
bash examples/perception/video_super_resolution/v3/build.sh
```

## Run

Default (generates sample test frames, 2x temporal + 4x spatial):

```bash
cvl run v3-vsr predict
```

Custom video (directory of PNG frames):

```bash
bash examples/perception/video_super_resolution/v3/predict.sh \
  --input /path/to/frames_dir \
  --space-scale 4 \
  --time-scale 8 \
  --output outputs/v3_result
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `vid4_city` | Input video directory or built-in sample name |
| `--output` | `outputs/v3_result` | Output directory for super-resolved frames |
| `--space-scale` | `4` | Spatial upsampling factor |
| `--time-scale` | `2` | Temporal upsampling factor |
| `--checkpoint` | (auto-download) | Path to V3 checkpoint file |
| `--cache-dir` | `~/.cache/cvlization/v3` | Cache directory for checkpoint |
| `--verbose` | off | Enable verbose logging |

## References

- Repository: https://github.com/prs-eth/v3
- Paper: https://arxiv.org/abs/2509.26325
- Checkpoint: https://drive.google.com/file/d/15nw5NhEIf7VvetEtQI1cnrLNWPi_9FGj
