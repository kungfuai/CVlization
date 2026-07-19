# LingBot-Video

MoE video foundation model for embodied intelligence from Robbyant.

## Overview

LingBot-Video is the first large-scale open-source Mixture-of-Experts (MoE) video
foundation model, trained on 70,000+ hours of embodied and web video data. It features
physical-rationality and task-completion reward alignment, making it suited for
generating videos with physically plausible dynamics.

Two model variants are available:

| Variant | Parameters | Active Params | Peak VRAM (81f T2V, bf16) | Download |
|---------|-----------|---------------|---------------------------|----------|
| Dense 1.3B | 1.3B | 1.3B | 22.9 GiB (23427 MiB) | ~5 GB |
| MoE 30B-A3B | 30B | ~3B | 78.1 GiB (79997 MiB) | ~121 GB |

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

**Dense 1.3B, T2I with structured prompt** (832x480, 40 steps):

![T2I Dense](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_dense_t2i_structured.png)

**MoE 30B-A3B, T2I with structured prompt** (832x480, 40 steps):

![T2I MoE](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_moe_t2i_structured.png)

**MoE 30B-A3B, T2V with structured prompt** (81 frames, 832x480, 40 steps) -- frames 0 / 40 / 80:

![T2V MoE frame 0](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_moe_t2v_structured_frame0.png)
![T2V MoE frame 40](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_moe_t2v_structured_frame40.png)
![T2V MoE frame 80](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_moe_t2v_structured_frame80.png)

Full video: [demo_moe_t2v_structured_81f.mp4](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_moe_t2v_structured_81f.mp4)

Note: MoE T2V with sequential CFG shows edge vignetting and painterly texture compared to dense. This may improve with batch_cfg (blocked, see Limitations).

**MoE 30B-A3B, TI2V with structured prompt** (41 frames, 832x480, 40 steps) -- frames 0 / 20 / 40:

![TI2V MoE frame 0](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_moe_ti2v_structured_frame0.png)
![TI2V MoE frame 20](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_moe_ti2v_structured_frame20.png)
![TI2V MoE frame 40](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_moe_ti2v_structured_frame40.png)

Full video: [demo_moe_ti2v_structured_41f.mp4](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_moe_ti2v_structured_41f.mp4)

Canonical TI2V input: [ti2v_first_frame_whiskey.png](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/ti2v_first_frame_whiskey.png) (upstream `assets/cases/ti2v/example_1/first_frame.png`)

## Structured vs Raw Prompts

The upstream DiT inference pipeline is designed for **structured JSON prompts** generated
by the Rewriter model (Qwen3-VL-27B). Raw natural-language prompts (`--prompt`) work but
produce lower quality output. For best results, use `--prompt-json` with a structured
caption file following the format in upstream `docs/en/dit_inference.md`.

Canonical structured prompts are hosted at:
- T2V/T2I: [canonical_t2v_prompt.json](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/canonical_t2v_prompt.json)
- T2I-specific: [canonical_t2i_prompt.json](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/canonical_t2i_prompt.json)
- TI2V: [canonical_ti2v_prompt.json](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/canonical_ti2v_prompt.json)

## Requirements

- **GPU**: NVIDIA GPU with 24 GB+ VRAM (dense T2V/TI2V) or 80 GB+ VRAM (MoE 81 frames)
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
| `--batch-cfg` | false | UNSUPPORTED: requires FA3 with SM120 kernels (no prebuilt wheel) |
| `--verbose` | false | Enable verbose logging |

## What to Expect

- **First run**: Downloads model weights from HuggingFace (~5 GB dense, ~121 GB MoE), cached afterward
- **Output**: MP4 video file (or PNG for T2I mode) in the current directory
- **Runtime on RTX PRO 6000 Blackwell (98 GB VRAM)**:
  - Dense T2V, 81 frames, 40 steps: ~2m24s
  - Dense T2I, 40 steps: ~4s
  - Dense TI2V, 41 frames, 40 steps: ~58s
  - MoE T2V, 81 frames, 40 steps: ~6m37s
  - MoE T2I, 40 steps: ~27s
  - MoE TI2V, 41 frames, 40 steps: ~3m10s
- **Peak VRAM** (measured with 200ms nvidia-smi polling, 832x480, sequential CFG, bf16):

  | Mode | Frames | Device Peak | Process Peak |
  |------|--------|-------------|--------------|
  | Dense T2V | 81 | 23427 MiB (22.9 GiB) | 23404 MiB |
  | Dense T2I | 1 | 16007 MiB (15.6 GiB) | 15984 MiB |
  | Dense TI2V | 41 | 23259 MiB (22.7 GiB) | 23236 MiB |
  | MoE T2V | 81 | 79997 MiB (78.1 GiB) | 79974 MiB |
  | MoE T2I | 1 | 70819 MiB (69.2 GiB) | 70796 MiB |
  | MoE TI2V | 41 | 78093 MiB (76.3 GiB) | 78070 MiB |

  GPU: RTX PRO 6000 Blackwell (97887 MiB total), idle baseline 15 MiB.
  Configuration: 5 denoising steps (cached run), seed 42, guidance 3.0, shift 3.0.
- **Resolution**: Default 832x480 (landscape). Both dimensions must be multiples of 16
- **Duration**: Default 81 frames at 24 FPS = ~3.4 seconds. If `--prompt-json` includes `duration`, frame count is derived automatically

## Limitations

- **Structured prompts recommended**: The model was trained on structured JSON captions from the Rewriter. Raw text prompts produce lower quality output. The Rewriter (Qwen3-VL-27B) is not included in this example.
- **MoE T2V edge vignetting**: MoE T2V with sequential CFG shows pillarbox edge bars and painterly texture. MoE TI2V does not exhibit this. The issue may be related to batch_cfg being unavailable (see below).
- **Refiner blocked**: The MoE checkpoint includes a refiner transformer (~60GB bf16) for upscaling base output. Tested: single GPU OOM (base ~94GB + refiner ~60GB > 98GB), FSDP across 2 GPUs crashes with CUDA memory error on SM120. The upstream runner preloads both models simultaneously with no sequential loading mode. See upstream `--run_refiner` for the workflow.
- **batch_cfg disabled**: The `--batch-cfg` flag requires `flash_attn_interface` (FlashAttention v3). No prebuilt FA3 wheel includes SM120 (Blackwell) kernels. FA3 `hopper/setup.py` targets SM80/SM90a/SM100a only. FA4 supports SM120 via JIT but uses a different API. Sequential CFG works correctly via PyTorch native SDPA.
- **MoE VRAM**: The MoE 30B-A3B variant requires ~78 GiB (79997 MiB) VRAM for 81-frame T2V at 832x480. T2I requires ~69 GiB (70819 MiB). Use fewer frames (`--num-frames 21`) to reduce memory.
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
