# MolmoAct2 — Vision-Language-Action Inference

## Why

MolmoAct2 is a 5B-parameter open Vision-Language-Action (VLA) model from Allen
AI that combines a Molmo2-ER VLM backbone with a flow-matching continuous action
expert for robot manipulation. It outperforms Pi-0.5 on LIBERO benchmarks and
the Think variant adds depth-token reasoning for richer spatial understanding
(Apache 2.0 license).

## What to Expect

- **First-run cost**: Downloads ~20 GB model checkpoint from HuggingFace on
  first run (cached at `~/.cache/huggingface/` afterward). Sample images are
  ~130 KB and download separately from `zzsi/cvl`.
- **What it does**: Given two camera images (agent-view + wrist) and a
  natural-language manipulation task, predicts a chunk of robot joint-pose
  actions using a flow-matching action expert.
- **Output location**: Saved to `molmoact2_outputs/` in your current working
  directory (when running via `cvl run`).
- **Output format**: `actions.npy` (NumPy array, shape `[chunk_len, action_dim]`)
  and `metrics.json` (chunk length, action dim, mean, std).
- **Runtime**: ~30–60 s on an A10 24 GB GPU (model load ~25 s, inference ~5 s).
  CUDA graph capture on first call adds ~10 s overhead.

## Sample

**Input** — agentview camera (auto-downloaded):

![Sample agentview](https://huggingface.co/datasets/zzsi/cvl/resolve/main/molmoact2/sample_agentview_rgb.png)

**Input** — wrist camera (auto-downloaded):

![Sample wrist](https://huggingface.co/datasets/zzsi/cvl/resolve/main/molmoact2/sample_wrist_rgb.png)

**Task**: `"put the white mug on the left plate and put the yellow and white mug on the right plate"`

**Output** — predicted action chunk (`actions.npy`, all 10 steps):

```
t=  0: [-0.00504  -0.01218  -0.00330  -0.00072  +0.00440  -0.00066  -1.00000]
t=  1: [-0.00705  -0.00575  -0.00723  -0.00107  +0.00354  +0.00111  -1.00000]
t=  2: [-0.00632  -0.01052  -0.00344  -0.00085  +0.00272  +0.00074  -1.00000]
t=  3: [-0.00761  -0.02116  -0.00827  -0.00078  +0.00196  +0.00121  -1.00000]
t=  4: [-0.01270  -0.03959  -0.01409  -0.00112  +0.00184  +0.00246  -1.00000]
t=  5: [-0.01639  -0.09397  -0.02639  -0.00120  +0.00249  +0.00377  -1.00000]
t=  6: [-0.02413  -0.17508  -0.04540  -0.00031  +0.00346  +0.00299  -1.00000]
t=  7: [-0.02663  -0.24925  -0.06893  +0.00022  +0.00346  +0.00333  -0.99821]
t=  8: [-0.02750  -0.30965  -0.08452  +0.00028  +0.00267  +0.00359  -0.99894]
t=  9: [-0.02767  -0.36592  -0.10235  +0.00050  +0.00149  +0.00343  -1.00000]
```

`metrics.json`:
```json
{
  "action_chunk_length": 10,
  "action_dim": 7,
  "action_mean": -0.168,
  "action_std": 0.346
}
```

## Quick Start

```bash
# 1. Build
./build.sh

# 2. Run with sample images (downloads model ~20 GB on first run)
./predict.sh

# 3. Custom images and task
./predict.sh --images cam0.png cam1.png \
    --task "pick up the red block" \
    --state 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

# 4. Think variant with depth-token reasoning
./predict.sh --model allenai/MolmoAct2-Think-LIBERO \
    --norm-tag libero --enable-depth-reasoning
```

## Available Checkpoints

| Model ID | Cameras | State dim | Description |
|---|---|---|---|
| `allenai/MolmoAct2-LIBERO` | agentview, wrist | 8 | LIBERO sim (default) |
| `allenai/MolmoAct2-Think-LIBERO` | agentview, wrist | 8 | + depth reasoning |
| `allenai/MolmoAct2-DROID` | exterior_1, wrist | 8 | DROID real-world |
| `allenai/MolmoAct2-SO100_101` | top, side | 6 | SO-100/101 arms |
| `allenai/MolmoAct2-BimanualYAM` | top, left, right | 14 | Bimanual YAM |

## Options

```
--model               HuggingFace model ID (default: allenai/MolmoAct2-LIBERO)
--images              Paths to camera images (must match checkpoint camera order)
--task                Natural language task instruction
--state               Robot state vector (space-separated floats)
--norm-tag            Normalization tag (default: libero)
--action-mode         continuous | discrete (default: continuous)
--enable-depth-reasoning  Enable depth-token reasoning (Think variants)
--num-steps           Flow solver iterations (default: 10)
--device              auto | cuda | cpu (default: auto)
--output-dir          Output directory (default: molmoact2_outputs)
```

## Output

Outputs are saved to `molmoact2_outputs/` in the working directory:

- `actions.npy` — predicted action chunk, shape `[chunk_len, action_dim]`
- `metrics.json` — action statistics (chunk length, dim, mean, std)

## References

- Paper: https://arxiv.org/abs/2605.02881
- Code: https://github.com/allenai/molmoact2
- LeRobot fork: https://github.com/allenai/lerobot-molmoact2
- Models: https://huggingface.co/collections/allenai/molmoact2-models-69f81e05242e2499606b1be6
