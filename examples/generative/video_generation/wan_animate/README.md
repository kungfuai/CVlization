# Wan2.2 Animate (Dockerized)

Run the Wan-Animate model (animation + replacement) inside Docker.

## Requirements

- NVIDIA GPU (24GB+ VRAM recommended for 14B)
- Docker with GPU support
- Model weights downloaded locally

## Setup

### 1. Build the Docker image

```bash
bash examples/generative/video_generation/wan_animate/build.sh
```

## Usage

The prediction script runs preprocessing + generation in one call.
Model weights are downloaded lazily on first run.

Weights are stored under:

```
~/.cache/cvlization/models/wan_animate/Wan2.2-Animate-14B
```

You can override this location with `WAN_ANIMATE_MODELS_DIR`.

Optional prefetch:

```bash
bash examples/generative/video_generation/wan_animate/download_models.sh
```

### Animation mode

```bash
export CUDA_VISIBLE_DEVICES=1
bash examples/generative/video_generation/wan_animate/predict.sh \
  --mode animate \
  --video /user_data/path/to/video.mp4 \
  --reference-image /user_data/path/to/image.jpg \
  --output-dir outputs/wan_animate_out
```

### Replacement mode

```bash
export CUDA_VISIBLE_DEVICES=1
bash examples/generative/video_generation/wan_animate/predict.sh \
  --mode replace \
  --video /user_data/path/to/video.mp4 \
  --reference-image /user_data/path/to/image.jpg \
  --output-dir outputs/wan_animate_out
```

## Path behavior

- Standalone mode: your current directory is mounted to `/user_data` (read-only).
  - Use `/user_data/...` for inputs.
  - Default outputs go to `/workspace/outputs` unless you pass an output path.
- CVL mode: your working directory is mounted to `/mnt/cvl/workspace`.
  - Relative paths resolve to `/mnt/cvl/workspace`.

## Notes

- `CUDA_VISIBLE_DEVICES=1` is the default unless you override it.
- Preprocessing output is saved under `<output-dir>/process_results`.
- If you already ran preprocessing, you can re-run generation only:
  - `--skip-preprocess --generate-args ...`
- If you only want preprocessing:
  - `--skip-generate`
