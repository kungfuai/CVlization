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

**MoE 30B-A3B, T2V** (21 frames, 832x480, 40 steps) -- frame 10:

![T2V MoE sample frame](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_t2v_moe_21f_frame10.png)

**MoE 30B-A3B, T2I** (832x480, 40 steps):

![T2I MoE sample](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_t2i_moe.png)

**Dense 1.3B, T2V** (81 frames, 832x480, 40 steps) -- frame 40:

![T2V Dense sample frame](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/demo_t2v_dense_81f_frame40.png)

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

**Important**: This model produces the best results with detailed, descriptive prompts
(multiple sentences describing the scene, subject, clothing, camera angle, lighting).
Short prompts may produce poor or abstract output.

```bash
# Text-to-video with dense 1.3B model (default)
cvl run lingbot_video predict -- \
    --prompt "A young woman with long brown hair is standing in a bright apartment. She is wearing a cream cardigan over a white top, paired with beige trousers. The background has a sofa, plant, and large windows with soft natural light. The camera is stationary at eye level."

# Text-to-video with MoE 30B model (needs ~80GB VRAM for 81 frames)
cvl run lingbot_video predict -- --model moe-30b-a3b \
    --prompt "A young woman with long brown hair is standing in a bright apartment..."

# Text-to-image (MoE recommended for quality)
cvl run lingbot_video predict -- --model moe-30b-a3b --mode t2i \
    --prompt "A detailed scene of a woman in a modern apartment..." --output output.png

# Text+image-to-video (animate from a reference image)
cvl run lingbot_video predict -- --image reference.png \
    --prompt "The woman turns slightly and adjusts her cardigan"

# Fewer frames for faster inference
cvl run lingbot_video predict -- --num-frames 21 --steps 20
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | dense-1.3b | Model variant: `dense-1.3b` or `moe-30b-a3b` |
| `--mode` | t2v | Generation mode: `t2v` (video) or `t2i` (image) |
| `--prompt` | sample | Text prompt describing the video to generate |
| `--negative-prompt` | auto | Negative prompt (auto-loaded from package if empty) |
| `--image` | - | Conditioning image for text+image-to-video (TI2V) |
| `--output` | output.mp4 | Output file path |
| `--height` | 480 | Video height (multiple of 16) |
| `--width` | 832 | Video width (multiple of 16) |
| `--num-frames` | 81 | Frame count (must be 4n+1, e.g. 21, 41, 61, 81) |
| `--fps` | 24.0 | Output video frame rate |
| `--steps` | 40 | Denoising steps |
| `--guidance-scale` | 3.0 | CFG guidance scale |
| `--shift` | 3.0 | Flow matching timestep shift |
| `--seed` | 42 | Random seed |
| `--verbose` | false | Enable verbose logging |

## What to Expect

- **First run**: Downloads model weights from HuggingFace (~5 GB dense, ~121 GB MoE), cached afterward
- **Output**: MP4 video file (or PNG for T2I mode) in the current directory
- **Runtime on RTX PRO 6000 Blackwell (98 GB VRAM)**:
  - Dense 1.3B, 81 frames, 40 steps: ~2m24s
  - MoE 30B-A3B, 81 frames, 40 steps: ~6m27s
  - MoE 30B-A3B, 21 frames, 40 steps: ~1m28s
- **Peak VRAM**: Dense ~8 GB, MoE ~80 GB (81 frames at 832x480)
- **Resolution**: Default 832x480 (landscape). Both dimensions must be multiples of 16
- **Duration**: Default 81 frames at 24 FPS = ~3.4 seconds of video
- **Prompt quality**: Detailed, multi-sentence prompts produce far better results than short prompts

## Limitations

- **Detailed prompts required**: The model was trained on structured captions; short prompts produce poor results. Use multiple sentences describing scene, subject, camera, and lighting.
- **MoE VRAM**: The MoE 30B-A3B variant requires ~80 GB VRAM for 81-frame generation at 832x480. Use fewer frames (`--num-frames 21`) to reduce memory.
- **MoE download size**: The MoE model is ~121 GB (includes transformer + refiner shards). Ensure sufficient disk space.
- **No prompt rewriter**: The upstream prompt rewriter (Qwen3-VL-27B) is not included; plain text prompts are used directly.
- **T2I quality varies**: T2I mode works best with MoE; dense 1.3B produces lower quality single images.

## Links

- [GitHub](https://github.com/Robbyant/lingbot-video)
- [HuggingFace (Dense 1.3B)](https://huggingface.co/robbyant/lingbot-video-dense-1.3b)
- [HuggingFace (MoE 30B)](https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b)
- [Paper](https://huggingface.co/papers/2607.07675)
