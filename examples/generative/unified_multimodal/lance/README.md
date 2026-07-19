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

## What to Expect

- **First run**: Downloads ~39 GB of model weights (cached afterward in
  `~/.cache/huggingface/`). A ~1 MB sample image is also downloaded for
  default VQA/edit demos.
- **Task**: Default mode is text-to-image (768x768 PNG). VQA and image
  editing use a canonical sample image when `--input-image` is omitted.
- **Output location**: Saved to `--output-dir` (default: `./artifacts`).
  E.g., `artifacts/000000.png`, `artifacts/metrics.json`.
- **Runtime**: ~18s model loading + ~7s inference at 30 steps on an
  RTX PRO 6000 Blackwell. Use `--num-steps 5` for fast validation (~2.5s
  inference).

## Sample

**Text-to-image** — prompt: *"A golden retriever sitting in a sunlit meadow
with wildflowers"*

![t2i output](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lance/t2i_output.png)

**Image editing** — instruction: *"Add a red collar to the dog"* (input:
same image above)

![edit output](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lance/edit_output.png)

**Image understanding** — question: *"What animal is in this image?"* —
answer: **Dog**
([full result](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lance/vqa_result.json))

## What

| Task | Flag | Description |
|------|------|-------------|
| Text-to-image | `--task t2i` | Generate a 768x768 image from a text prompt |
| Image understanding | `--task x2t_image` | Answer a question about an image |
| Image editing | `--task image_edit` | Edit an image given a text instruction |

Model weights are downloaded automatically from
[bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance)
on first run (~39 GB, cached afterward).

## Quick Start

```bash
# Build
./build.sh

# Text-to-image (default)
./predict.sh --task t2i --prompt "A cat wearing a top hat, oil painting style"

# Image understanding (uses canonical sample image by default)
./predict.sh --task x2t_image --prompt "What objects are in this image?"

# Image editing (uses canonical sample image by default)
./predict.sh --task image_edit --edit-instruction "Add sunglasses"

# With a custom input image
./predict.sh --task x2t_image --input-image photo.jpg --prompt "Describe this scene"
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | `t2i` | Task: `t2i`, `x2t_image`, `image_edit` |
| `--prompt` | *(sunset scene)* | Text prompt or VQA question |
| `--input-image` | *(canonical sample)* | Input image (auto-downloaded for VQA/edit) |
| `--edit-instruction` | *"Add a red collar to the dog"* | Edit instruction |
| `--output-dir` | `./artifacts` | Where outputs are saved |
| `--model-id` | `bytedance-research/Lance` | HuggingFace model repo |
| `--resolution` | `768` | Image resolution (pixels) |
| `--num-steps` | `30` | Denoising steps |
| `--cfg-scale` | `4.0` | Classifier-free guidance scale |

## Output

- **t2i**: PNG image (e.g., `000000.png`) saved to `--output-dir`
- **x2t_image**: JSON with question-answer pairs (`result.json`) in `--output-dir`
- **image_edit**: Edited PNG image saved to `--output-dir`
- All tasks produce `metrics.json` summarizing the run.

## Hardware Requirements

- **GPU**: NVIDIA GPU with >= 40 GB VRAM (A100, A6000, etc.) —
  *unverified upstream estimate; tested on 98 GB only*
- **Disk**: ~39 GB for model weights + Docker image
- **Note**: The 3B model may fit on 24 GB GPUs for image-only tasks at
  lower resolution, but this is not officially supported by upstream.

## References

- Paper: [Lance: Unified Multimodal Modeling by Multi-Task Synergy](https://huggingface.co/papers/2605.18678)
- Code: [github.com/bytedance/Lance](https://github.com/bytedance/Lance)
- Weights: [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance) (Apache 2.0)
- Project page: [lance-project.github.io](https://lance-project.github.io/)
