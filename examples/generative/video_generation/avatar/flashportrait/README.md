# FlashPortrait

6x faster infinite portrait animation using Wan2.1-14B Video DiT.

**Paper**: [arXiv:2512.16900](http://arxiv.org/abs/2512.16900)
**Original repo**: [Francis-Rings/FlashPortrait](https://github.com/Francis-Rings/FlashPortrait)

## Features

- **6x faster** than comparable diffusion-based portrait animation
- **Infinite-length** video generation with sliding window
- **ID-preserving** facial animation
- **Multiple resolutions**: 512x512, 480x832, 832x480, 720x720, 720x1280, 1280x720

## Requirements

- **GPU**: 60-80GB VRAM for model_full_load, ~18GB for sequential_cpu_offload
- **Disk**: ~70GB for model weights (Wan2.1-14B + FlashPortrait)
- **RAM**: ~100GB+ for model loading
- **Flash Attention**: Required for stable execution (installed via pre-built wheel)

## Quick Start

```bash
# Build
cvl run flashportrait build

# Run with sample inputs (auto-downloaded from HuggingFace)
cvl run flashportrait predict \
    --image test_inputs/reference.png \
    --video test_inputs/driven_video.mp4 \
    --output outputs/result.mp4 \
    --fast

# Run with your own inputs
cvl run flashportrait predict \
    --image your_portrait.jpg \
    --video your_driving.mp4 \
    --output output.mp4 \
    --fast

# Higher resolution (slower)
cvl run flashportrait predict \
    --image test_inputs/reference.png \
    --video test_inputs/driven_video.mp4 \
    --output outputs/result_720.mp4 \
    --fast \
    --max_size 720
```

Sample inputs are automatically downloaded from [zzsi/cvl](https://huggingface.co/datasets/zzsi/cvl/tree/main/flashportrait) on first run.

## Performance

Tested on NVIDIA RTX PRO 6000 (98GB VRAM), 297 frames (~12s video):

| Resolution | Mode | Time | VRAM |
|------------|------|------|------|
| 512x512 | fast (4 steps) | ~8 min | ~60GB |
| 720x720 | fast (4 steps) | ~22 min | ~80GB |

## GPU Memory Modes

| Mode | VRAM | Speed | Notes |
|------|------|-------|-------|
| `model_full_load` | ~60-80GB | Fastest | |
| `model_cpu_offload` | ~30-40GB | Medium | |
| `sequential_cpu_offload` | ~18GB | Slowest | Tested on A100, works on RTX 3090 |
| `model_cpu_offload_and_qfloat8` | ~20-30GB | Medium | FP8 quantization |

## Parameters

- `--image`: Reference portrait image
- `--video`: Driving video for motion transfer
- `--output`: Output video path
- `--fast`: Enable 4-step inference with LoRA + tiny VAE
- `--steps`: Inference steps (default: 4 with --fast, 30 otherwise)
- `--max_size`: Max resolution dimension (default: 512)
- `--gpu_mode`: Memory mode (default: model_cpu_offload)
- `--cfg_scale`: Guidance scale (default: 4.0)
- `--emo_cfg_scale`: Emotion CFG scale (default: 4.0)

## License

Apache 2.0 (same as original FlashPortrait)
