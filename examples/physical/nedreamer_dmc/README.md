# NE-Dreamer on DeepMind Control Suite

Model-based RL agent from ["Next Embedding Prediction Makes World Models Stronger"](https://arxiv.org/abs/2603.02765). NE-Dreamer replaces the decoder in DreamerV3 with a temporal transformer that predicts next-step encoder embeddings, learning state representations without pixel reconstruction.

## Quick Start

```bash
./build.sh
./train.sh                           # default: walker_walk, 10k steps, 12M model
TASK=dmc_cheetah_run ./train.sh      # different task
STEPS=1000000 ./train.sh             # full training run
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TASK` | `dmc_walker_walk` | DMC task name |
| `STEPS` | `10000` | Total environment steps |
| `SEED` | `0` | Random seed |
| `ENV_NUM` | `4` | Number of parallel training environments |
| `EVAL_EPISODES` | `2` | Number of evaluation episodes |
| `DEVICE` | `cuda:0` | Device |
| `WANDB_MODE` | `disabled` | Set to `online` to enable wandb logging |

## Algorithm Selection

Pass `model.rep_loss=...` as extra arg to `train.sh`:

| Algorithm | Argument |
|-----------|----------|
| NE-Dreamer (default) | `model.rep_loss=ne_dreamer` |
| DreamerV3 | `model.rep_loss=dreamer` |
| R2-Dreamer | `model.rep_loss=r2dreamer` |
| DreamerPro | `model.rep_loss=dreamerpro` |

## Model Sizes

Pass `model=size...` as extra arg:

| Config | Params | VRAM |
|--------|--------|------|
| `size12M` (default) | ~12M | ~4 GB |
| `size25M` | ~25M | ~6 GB |
| `size50M` | ~50M | ~10 GB |
| `size100M` | ~100M | ~16 GB |
| `size200M` | ~200M | ~20 GB |

## References

- Paper: https://arxiv.org/abs/2603.02765
- Source: https://github.com/corl-team/nedreamer
- License: MIT
