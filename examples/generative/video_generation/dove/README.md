# DOVE — One-Step Video Super-Resolution

CVL wrapper for **DOVE** (NeurIPS 2025): a one-step diffusion model for real-world video super-resolution built on CogVideoX-1.5-5B. Upscales low-resolution video up to 4× in a single forward pass, achieving up to 28× speed-up over multi-step diffusion methods.

> **Attribution:** This example wraps the official DOVE implementation.
> Paper: [DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](https://arxiv.org/abs/2505.16239) (NeurIPS 2025)
> Code: [github.com/zhengchen1999/DOVE](https://github.com/zhengchen1999/DOVE)
> Authors: Zheng Chen, Zichen Zou, Kewei Zhang, Xiongfei Su, Xin Yuan, Yong Guo, Yulun Zhang

## Sample

**Input** — low-resolution clip (auto-downloaded from `zzsi/cvl`):

![input](https://huggingface.co/datasets/zzsi/cvl/resolve/main/dove/sample_input.gif)

**Output** — 4× upscaled result:

![output](https://huggingface.co/datasets/zzsi/cvl/resolve/main/dove/sample_output.gif)

## What to Expect

- **First run**: downloads ~40GB (DOVE/CogVideoX weights from `zzsi/DOVE`), cached to `~/.cache/huggingface/` afterward
- **What it does**: upscales each frame of the input video using one-step diffusion; default 4× upscale (e.g. 256×256 → 1024×1024)
- **Output location**: saved to `outputs/dove_out.mp4` in your current working directory
- **Output format**: MP4 at the upscaled resolution, same FPS as input
- **Runtime**: ~40s for a 100-frame 256×256 clip on an A100/H100 class GPU; requires ~43GB VRAM (with VAE slicing/tiling)

## Requirements

- NVIDIA GPU with >= 40GB VRAM
- Docker with NVIDIA runtime
- Internet access for model download

## Build

```bash
bash examples/generative/video_generation/dove/build.sh
```

## Run

Default (sample input, 4× upscale):

```bash
cvl run dove predict
```

Custom video:

```bash
bash examples/generative/video_generation/dove/predict.sh \
  --input path/to/video.mp4 \
  --output outputs/upscaled.mp4 \
  --upscale 4
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `sample` | Input video path or `sample` |
| `--output` | `outputs/dove_out.mp4` | Output video path |
| `--upscale` | `4` | Upscale factor |
| `--dtype` | `bfloat16` | Model dtype (`float16`, `bfloat16`, `float32`) |
| `--is_vae_st` | off | Enable VAE slicing+tiling (saves VRAM) |
| `--chunk_len` | `0` | Temporal chunk length (0 = process all frames at once) |
| `--tile_size_hw` | `0 0` | Spatial tile size H W (0 0 = no tiling) |
| `--dry-run` | — | Validate config and exit |

## Citation

```bibtex
@inproceedings{chen2025dove,
  title={DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution},
  author={Chen, Zheng and Zou, Zichen and Zhang, Kewei and Su, Xiongfei and Yuan, Xin and Guo, Yong and Zhang, Yulun},
  booktitle={NeurIPS},
  year={2025}
}
```
