# Lance — Unified Multimodal Inference

A single 3B-parameter model that handles **text-to-image generation**,
**image understanding (VQA)**, and **image editing** within one unified
architecture.

## Why

Most multimodal pipelines chain separate models for understanding and
generation. Lance (ByteDance Research, 2025) trains a single LLM backbone
(Qwen2.5-VL-3B) to do both — plus editing — by treating all tasks as
interleaved token-sequence problems. This example lets you exercise all
three capabilities from one Docker container.

## What

| Task | Flag | Description |
|------|------|-------------|
| Text-to-image | `--task t2i` | Generate a 768x768 image from a text prompt |
| Image understanding | `--task x2t_image` | Answer a question about an image |
| Image editing | `--task image_edit` | Edit an image given a text instruction |

Model weights are downloaded automatically from
[bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance)
on first run (~25 GB for image-only weights).

## What to Expect

- **First run**: Downloads ~25 GB of model weights (cached afterward in
  `~/.cache/huggingface/`).
- **Task**: Generates a 768x768 PNG image from a text prompt (default t2i mode).
- **Output location**: Saved to your current working directory (e.g., `000000.png`,
  `metrics.json`).
- **Runtime**: ~18s model loading + ~15s inference at 30 steps on an RTX PRO 6000.
  Use `--num-steps 5` for fast validation (~3s inference).

## Sample

**Input** (text prompt):
```
A cat
```

**Output** — 768x768 PNG image (`000000.png`), generated in ~2.4s with `--num-steps 5`.

## Quick Start

```bash
# Build
./build.sh

# Text-to-image (default)
./predict.sh --task t2i --prompt "A cat wearing a top hat, oil painting style"

# Image understanding
./predict.sh --task x2t_image \
  --input-image /path/to/photo.jpg \
  --prompt "What objects are in this image?"

# Image editing
./predict.sh --task image_edit \
  --input-image /path/to/photo.jpg \
  --edit-instruction "Add a rainbow in the sky"
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | `t2i` | Task: `t2i`, `x2t_image`, `image_edit` |
| `--prompt` | *(sunset scene)* | Text prompt or VQA question |
| `--input-image` | — | Path to input image (required for x2t_image, image_edit) |
| `--edit-instruction` | — | Edit instruction (required for image_edit) |
| `--output-dir` | `./artifacts` | Where outputs are saved |
| `--model-id` | `bytedance-research/Lance` | HuggingFace model repo |
| `--resolution` | `768` | Image resolution (pixels) |
| `--num-steps` | `30` | Denoising steps |
| `--cfg-scale` | `4.0` | Classifier-free guidance scale |

## Output

- **t2i**: PNG image (e.g., `000000.png`) saved to current working directory
- **x2t_image**: JSON with question-answer pairs (`result.json`)
- **image_edit**: Edited PNG image saved to current working directory
- All tasks produce `metrics.json` summarizing the run.

## Hardware Requirements

- **GPU**: NVIDIA GPU with >= 40 GB VRAM (A100, A6000, etc.)
- **Disk**: ~30 GB for model weights + Docker image
- **Note**: The 3B model may fit on 24 GB GPUs for image-only tasks at
  lower resolution, but this is not officially supported by upstream.

## References

- Paper: [Lance: Unified Multimodal Modeling by Multi-Task Synergy](https://huggingface.co/papers/2605.18678)
- Code: [github.com/bytedance/Lance](https://github.com/bytedance/Lance)
- Weights: [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance) (Apache 2.0)
- Project page: [lance-project.github.io](https://lance-project.github.io/)
