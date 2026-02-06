# FlashVSR (v1.1)

FlashVSR is a diffusion-based streaming video super-resolution model. This example runs **v1.1** with the **tiny** pipeline by default and supports the full pipeline via `--mode full`.

## Requirements

- NVIDIA GPU with **>= 80GB VRAM**
  - Recommended: A100 / A800 / H200
  - Other GPUs are unverified and may fail or be unstable
- Docker with GPU support
- Internet access to download model weights

## Build

```bash
bash examples/generative/video_generation/flashvsr/build.sh
```

## Predict (standalone)

```bash
bash examples/generative/video_generation/flashvsr/predict.sh \
  --output outputs/flashvsr_out.mp4
```

The default `--input` is a sample video downloaded from `zzsi/cvl` (dataset) at `flashvsr/example0.mp4`.

Use the full pipeline:

```bash
bash examples/generative/video_generation/flashvsr/predict.sh \
  --mode full \
  --output outputs/flashvsr_out.mp4
```

## Predict (CVL)

```bash
cvl run flashvsr predict --input input.mp4 --output outputs/flashvsr_out.mp4
```

## Options

- `--mode tiny|full` (default: `tiny`)
- `--input` (default: `sample`) accepts `sample` or `example0` for the hosted test video
- `--scale` (default: `4.0`)
- `--sparse_ratio` (default: `2.0`)
- `--local_range` (default: `11`)
- `--weights_dir` (optional): path to a pre-downloaded `FlashVSR-v1.1` folder
- `--dry-run`: validate environment and weights, then exit

## Notes

- Weights are downloaded from Hugging Face: `JunhaoZhuang/FlashVSR-v1.1`.
- Block-Sparse-Attention is required and compiled during Docker build.
- This example is optimized for **4x super-resolution**.

## References

- FlashVSR repo: https://github.com/OpenImagingLab/FlashVSR
- Block-Sparse-Attention: https://github.com/mit-han-lab/Block-Sparse-Attention
- Model weights (v1.1): https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1
