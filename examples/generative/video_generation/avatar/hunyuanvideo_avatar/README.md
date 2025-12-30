# HunyuanVideo-Avatar (Dockerized Example)

HunyuanVideo-Avatar generates audio-driven avatar videos from a reference image and a driving audio clip.

## Important License and Territory Restrictions

This example uses Tencent HunyuanVideo-Avatar, which is **not** licensed for use in the EU, UK, or South Korea.
See `examples/generative/video_generation/avatar/hunyuanvideo_avatar/LICENSE.txt` and
`examples/generative/video_generation/avatar/hunyuanvideo_avatar/NOTICE.txt` for details.

## Requirements

- NVIDIA GPU with CUDA support
- Docker with NVIDIA runtime
- Disk: large model download (tens of GB)

## Build

```bash
cvl run hunyuanvideo_avatar build
```

## Run Inference

```bash
cvl run hunyuanvideo_avatar predict \
  --image /path/to/reference.png \
  --audio /path/to/driving.wav \
  --output outputs/result.mp4
```

Sample inputs are automatically downloaded from
`zzsi/cvl` (dataset) under `hunyuan_avatar/` when `--image`/`--audio` are omitted
(`1.png` and `2.WAV`).

### Useful Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | "A person is speaking." | Text prompt for scene |
| `--image-size` | 704 | Target square size |
| `--infer-steps` | 50 | Denoising steps |
| `--cfg-scale` | 7.5 | Guidance scale |
| `--cpu-offload` | off | Lower VRAM usage, slower |
| `--infer-min` | off | Force short (~5s) output |
| `--no-fp8` | off | Use FP16 checkpoint (larger VRAM) |

## Model Weights

Weights are downloaded on first run to:
`~/.cache/cvlization/hunyuanvideo_avatar/weights`

## References

- https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar
- https://arxiv.org/abs/2505.20156
