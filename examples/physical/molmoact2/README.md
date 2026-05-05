# MolmoAct2 — Vision-Language-Action Inference

## Why

MolmoAct2 is a 5B-parameter open Vision-Language-Action (VLA) model from
Allen AI that combines a Molmo2-ER VLM backbone with a flow-matching
continuous action expert for robot manipulation. It outperforms previous
open VLA models on LIBERO, DROID, and bimanual manipulation benchmarks,
and the Think variant adds depth-token reasoning for richer spatial
understanding.

## What

This example runs **inference-only** with a pre-trained MolmoAct2
checkpoint. Given camera images, a task instruction, and robot state, it
predicts an action chunk that can be sent directly to a robot controller.

Available checkpoints:

| Checkpoint | Cameras | State dim | Description |
|---|---|---|---|
| `allenai/MolmoAct2-LIBERO` | agentview, wrist | 8 | LIBERO sim benchmark |
| `allenai/MolmoAct2-Think-LIBERO` | agentview, wrist | 8 | + depth reasoning |
| `allenai/MolmoAct2-DROID` | exterior_1, wrist | 8 | DROID real-world |
| `allenai/MolmoAct2-SO100_101` | top, side | 6 | SO-100/101 arms |
| `allenai/MolmoAct2-BimanualYAM` | top, left, right | 14 | Bimanual YAM |

## Quick Start

```bash
# 1. Build
./build.sh

# 2. Run with sample images from checkpoint
./predict.sh

# 3. Custom images and task
./predict.sh --images cam0.png cam1.png \
    --task "pick up the red block" \
    --state 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

# 4. Think variant
./predict.sh --model allenai/MolmoAct2-Think-LIBERO \
    --enable-depth-reasoning
```

## Options

```
--model             HuggingFace model ID (default: allenai/MolmoAct2-LIBERO)
--images            Paths to camera images (must match checkpoint camera order)
--task              Natural language task instruction
--state             Robot state vector (space-separated floats)
--norm-tag          Normalization tag (default: libero)
--action-mode       continuous | discrete (default: continuous)
--enable-depth-reasoning  Enable depth-token reasoning (Think variants)
--num-steps         Flow solver iterations (default: 10)
--device            auto | cuda | cpu (default: auto)
--output-dir        Output directory (default: ./artifacts)
```

## Output

- `artifacts/actions.npy` — predicted action chunk as NumPy array
- `artifacts/metrics.json` — action statistics (chunk length, dim, mean, std)

## References

- Paper: https://arxiv.org/abs/2605.02881
- Code: https://github.com/allenai/molmoact2
- LeRobot fork: https://github.com/allenai/lerobot-molmoact2
- Models: https://huggingface.co/collections/allenai/molmoact2-models-69f81e05242e2499606b1be6
