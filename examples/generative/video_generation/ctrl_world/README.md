# Ctrl-World: Controllable World Model for Robot Manipulation

Action-conditioned video world model built on Stable Video Diffusion. Given an
initial robot observation and a sequence of actions (cartesian end-effector
poses), it predicts what the scene will look like at each future timestep.

This example runs **trajectory replay inference**: it takes recorded actions
from the DROID dataset and feeds them through the world model, producing a
side-by-side comparison of ground truth versus predicted video across three
camera views.

## Quick start

```bash
# Build
./build.sh

# Run default replay (1 trajectory, 12 interaction steps)
./predict.sh

# Replay all 3 sample trajectories
./predict.sh --val_ids 899 18599 199

# Quick run with fewer interaction steps
./predict.sh --interact_num 3

# Smoke test
./test.sh
```

## What to expect

Running `cvl run ctrl_world predict` (or `./predict.sh`) will:

1. **Download models (~17 GB) and sample data (~3 MB)** on first run — cached
   in `~/.cache/huggingface/` for subsequent runs.
2. **Replay trajectory 899** through 12 autoregressive interaction steps
   (50 diffusion denoising steps each).
3. **Save an MP4 video** to `ctrl_world_outputs/` in your current working
   directory (e.g. `ctrl_world_outputs/replay_899_pick_up_the_..._20260218_143000.mp4`).

The output video shows a 6-panel grid at 4 fps: three camera views (exterior,
wrist, exterior-right) with ground truth on top and world model prediction on
the bottom for each view.

**Typical runtimes** (after model download):
- ~6 min on A100 (80 GB)
- ~3.5 min on H100
- ~10 s per interaction step on A100, ~5 s on H100

## Model weights

Downloaded automatically from HuggingFace on first run:

| Component | Source | Size |
|-----------|--------|------|
| SVD base model | `stabilityai/stable-video-diffusion-img2vid` | ~8 GB |
| Ctrl-World checkpoint | `yjguo/Ctrl-World` | ~8 GB |
| CLIP text encoder | `openai/clip-vit-base-patch32` | ~600 MB |

You can also point to local copies with `--svd_model_path`, `--clip_model_path`,
and `--ckpt_path`.

## Output

Each replay generates an MP4 video at 4 fps showing, for each of the three
camera views, ground truth (top) and world model prediction (bottom)
side-by-side. Output is saved to `ctrl_world_outputs/` in your working
directory by default. Use `--output <dir>` to change the output directory.

## Resource requirements

- GPU VRAM: ~24-32 GB (bfloat16 inference)
- Disk: ~20 GB for model weights (cached in HuggingFace cache)
- Speed: ~10 s per interaction step on A100, ~5 s on H100

## Sample data

A small subset of the DROID dataset (4 validation trajectories, ~3 MB) is
hosted on HuggingFace (`zzsi/cvl` dataset repo, `ctrl_world/` prefix) and
downloaded automatically on first run. Cached in
`~/.cache/huggingface/cvl_data/ctrl_world/`.

## References

- Paper: https://arxiv.org/abs/2510.10125
- Project page: https://ctrl-world.github.io/
- Source repo: https://github.com/Robert-gyj/Ctrl-World
- DROID dataset: https://huggingface.co/datasets/cadene/droid_1.0.1
