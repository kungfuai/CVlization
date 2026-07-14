# LingBot-Video

MoE video foundation model for embodied intelligence from Robbyant.

## Overview

LingBot-Video is the first large-scale open-source Mixture-of-Experts (MoE) video
foundation model, trained on 70,000+ hours of embodied and web video data. It features
physical-rationality and task-completion reward alignment, making it suited for
generating videos with physically plausible dynamics.

Two model variants are available:

| Variant | Parameters | Active Params | Peak VRAM (81 frames, bf16) | Download |
|---------|-----------|---------------|---------------------------|----------|
| Dense 1.3B | 1.3B | 1.3B | ~8 GB | ~5 GB |
| MoE 30B-A3B | 30B | ~3B | ~80 GB | ~121 GB |

## Sample Output

**Dense 1.3B, T2V with structured prompt** (81 frames, 832x480, 40 steps) -- frames 0 / 40 / 80:

![T2V Dense frame 0](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_t2v_dense_structured_frame0.png)
![T2V Dense frame 40](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_t2v_dense_structured_frame40.png)
![T2V Dense frame 80](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_t2v_dense_structured_frame80.png)

Full video: [demo_t2v_dense_structured_81f.mp4](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_t2v_dense_structured_81f.mp4)

**Dense 1.3B, TI2V with structured prompt** (41 frames, 832x480, 40 steps) -- frames 0 / 20 / 40:

![TI2V Dense frame 0](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_ti2v_dense_structured_frame0.png)
![TI2V Dense frame 20](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_ti2v_dense_structured_frame20.png)
![TI2V Dense frame 40](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_ti2v_dense_structured_frame40.png)

Full video: [demo_ti2v_dense_structured_41f.mp4](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_ti2v_dense_structured_41f.mp4)

Canonical TI2V input image: [ti2v_input.png](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/ti2v_input.png)

## Structured vs Raw Prompts

The upstream DiT inference pipeline is designed for **structured JSON prompts** generated
by the Rewriter model (Qwen3-VL-27B). Raw natural-language prompts (`--prompt`) work but
produce lower quality output. For best results, use `--prompt-json` with a structured
caption file following the format in upstream `docs/en/dit_inference.md`.

A canonical structured prompt is hosted at:
[zzsi/cvl/lingbot_video/canonical_t2v_prompt.json](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/canonical_t2v_prompt.json)

## Requirements

- **GPU**: NVIDIA GPU with 8 GB+ VRAM (dense) or 80 GB+ VRAM (MoE 81 frames)
- **Disk**: ~5 GB for dense weights, ~121 GB for MoE weights (downloaded on first run)
- **Docker**: With NVIDIA runtime

## Usage

### Build

```bash
cvl run lingbot_video build
```

### Run Inference

**With structured prompt (recommended)**:

```bash
# Download the canonical structured prompt
curl -L -o canonical_t2v_prompt.json \
    https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/canonical_t2v_prompt.json

# Text-to-video with structured prompt
cvl run lingbot_video predict -- \
    --prompt-json canonical_t2v_prompt.json

# Text+image-to-video (animate from a reference image)
# Download the canonical TI2V input image:
curl -L -o ti2v_input.png \
    https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/ti2v_input.png
cvl run lingbot_video predict -- \
    --prompt-json canonical_t2v_prompt.json --image ti2v_input.png

# MoE variant (needs ~80GB VRAM)
cvl run lingbot_video predict -- --model moe-30b-a3b \
    --prompt-json canonical_t2v_prompt.json
```

**With raw text prompt (best-effort)**:

```bash
# Raw text prompts work but produce lower quality than structured prompts.
# Use detailed, multi-sentence descriptions for best results.
cvl run lingbot_video predict -- \
    --prompt "A young woman with long brown hair standing in a bright apartment. She is wearing a cream cardigan over a white top, paired with beige trousers. The background has a sofa, plant, and large windows with soft natural light. The camera is stationary at eye level."

# Text-to-image (MoE recommended)
cvl run lingbot_video predict -- --model moe-30b-a3b --mode t2i \
    --prompt "A detailed scene description..." --output output.png

# Fewer frames for faster inference
cvl run lingbot_video predict -- --num-frames 21 --steps 20 \
    --prompt "A detailed scene description..."
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | dense-1.3b | Model variant: `dense-1.3b` or `moe-30b-a3b` |
| `--mode` | t2v | Generation mode: `t2v` (video) or `t2i` (image) |
| `--prompt` | - | Raw text prompt (best-effort quality without Rewriter) |
| `--prompt-json` | - | Structured prompt JSON file (official format, best quality) |
| `--negative-prompt` | auto | Negative prompt (auto-loaded from package if empty) |
| `--image` | - | Conditioning image for text+image-to-video (TI2V) |
| `--output` | output.mp4 | Output file path |
| `--height` | 480 | Video height (multiple of 16) |
| `--width` | 832 | Video width (multiple of 16) |
| `--num-frames` | 81 | Frame count (4n+1). If omitted, derived from `duration` in `--prompt-json` |
| `--fps` | 24.0 | Output video frame rate |
| `--steps` | 40 | Denoising steps |
| `--guidance-scale` | 3.0 | CFG guidance scale |
| `--shift` | 3.0 | Flow matching timestep shift |
| `--seed` | 42 | Random seed |
| `--batch-cfg` | false | Batch CFG (requires FlashAttention v3 / Hopper+) |
| `--verbose` | false | Enable verbose logging |

## What to Expect

- **First run**: Downloads model weights from HuggingFace (~5 GB dense, ~121 GB MoE), cached afterward
- **Output**: MP4 video file (or PNG for T2I mode) in the current directory
- **Runtime on RTX PRO 6000 Blackwell (98 GB VRAM)**:
  - Dense 1.3B, 81 frames, 40 steps: ~2m24s
  - Dense 1.3B, 41 frames (TI2V), 40 steps: ~58s
  - MoE 30B-A3B, 81 frames, 40 steps: ~6m27s
- **Peak VRAM**: Dense ~8 GB, MoE ~80 GB (81 frames at 832x480)
- **Resolution**: Default 832x480 (landscape). Both dimensions must be multiples of 16
- **Duration**: Default 81 frames at 24 FPS = ~3.4 seconds of video

## Limitations

- **Structured prompts recommended**: The model was trained on structured JSON captions from the Rewriter. Raw text prompts produce lower quality output. The Rewriter (Qwen3-VL-27B) is not included in this example.
- **Refiner not exposed**: The MoE checkpoint includes a refiner transformer for upscaling base output to higher resolution. This example verifies base generation only; the refiner path is not exposed. See upstream `scripts/inference.py --run_refiner` for the full refiner workflow.
- **batch_cfg blocked**: The `--batch-cfg` flag requires `flash_attn_interface` (FlashAttention v3 / Hopper). Prebuilt wheels (sm80/sm90a/sm100a) do not cover sm120 (Blackwell Max-Q). Sequential CFG works correctly.
- **T2I quality varies**: T2I mode works best with MoE; dense 1.3B produces lower quality single images.
- **MoE VRAM**: The MoE 30B-A3B variant requires ~80 GB VRAM for 81-frame generation at 832x480. Use fewer frames (`--num-frames 21`) to reduce memory.
- **MoE download size**: The MoE model is ~121 GB (includes transformer + refiner shards). Ensure sufficient disk space.

## Pinned Dependencies

Upstream pinned to commit [`a638721`](https://github.com/Robbyant/lingbot-video/commit/a638721cf2271804d02738b69f2ad788c4a559fc).

Resolved dependency stack in Docker image:
- torch 2.7.0+cu128
- diffusers 0.39.0
- transformers 5.13.1
- accelerate 1.14.0
- peft 0.19.1

FlashAttention is not installed. The wrapper uses PyTorch native SDPA for
sequential CFG attention. The upstream runner (`lingbot_video.runner`) can be
invoked with `LINGBOT_QWEN_ATTN_IMPLEMENTATION=sdpa` to bypass its FA3 default.

## Links

- [GitHub](https://github.com/Robbyant/lingbot-video)
- [HuggingFace (Dense 1.3B)](https://huggingface.co/robbyant/lingbot-video-dense-1.3b)
- [HuggingFace (MoE 30B)](https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b)
- [Paper](https://huggingface.co/papers/2607.07675)
