# Krea Realtime Video (via Scope)

Working implementation of Krea Realtime 14B using the [Scope framework](https://github.com/daydreamlive/scope).

## Why This Example?

The official Krea SDK (`krea-realtime` example) does not yet support the 14B model architecture - it has the 14B architecture option commented out and only loads the 1.3B architecture. This example uses the **Scope framework's production-ready implementation** which properly handles 14B models.

## Model Details

- **Size**: 14 billion parameters (28.6GB safetensors file)
- **Base**: Krea AI's distilled real-time video model
- **Performance**: Optimized for real-time generation with 4-6 inference steps
- **Framework**: Uses Scope's implementation with proper 14B architecture support

## Requirements

- **GPU**: NVIDIA GPU with 40GB+ VRAM recommended
  - H100, RTX 6000 Pro, A100 recommended
  - Can use 32GB GPUs (RTX A6000) with FP8 quantization
- **Docker** with NVIDIA Container Toolkit
- **Disk**: 50GB space (models + Scope framework)

## Quick Start

### 1. Build the Docker image

```bash
./build.sh
```

This builds a Docker image with Scope framework and dependencies (~30GB).

### 2. Generate video from text

```bash
PROMPT="A serene mountain lake at sunset with reflections" ./predict.sh
```

Output will be saved to `outputs/video.mp4`. Customize generation:

```bash
PROMPT="A person walking through a forest" \
WIDTH=832 \
HEIGHT=480 \
NUM_BLOCKS=9 \
SEED=42 \
./predict.sh
```

## Usage with CVL CLI

```bash
# Build the container
cvl run krea-realtime-scope build

# Generate video from text
PROMPT="A beautiful landscape" cvl run krea-realtime-scope predict
```

## Advanced Options

### Offline Batch Generation

Run inside the Docker container or customize via environment variables:

```bash
python3 predict.py \
    --prompt "Your prompt here" \
    --output outputs/custom.mp4 \
    --width 832 \
    --height 480 \
    --num-blocks 9 \
    --seed 42 \
    --fps 24 \
    --quantization none
```

**Parameters**:
- `--prompt`: Text description of desired video (required)
- `--output`: Output video path (default: outputs/video.mp4)
- `--width`: Video width in pixels (default: 832)
- `--height`: Video height in pixels (default: 480)
- `--num-blocks`: Number of blocks to generate - controls video length (default: 9)
- `--seed`: Random seed for reproducibility (default: 42)
- `--fps`: Frames per second for output video (default: 24)
- `--quantization`: Use "fp8" for 32GB GPUs, "none" for 40GB+ GPUs (default: none)

## Differences from krea-realtime Example

| Feature | krea-realtime | krea-realtime-scope |
|---------|---------------|---------------------|
| **Status** | ❌ Blocked (SDK issue) | ✅ Working |
| **SDK/Framework** | Official Krea SDK | Scope framework |
| **14B Support** | ❌ Not implemented | ✅ Full support |
| **VRAM** | 40GB+ required | 32GB+ (with FP8) |
| **Maintenance** | Waits for Krea | Tracks Scope updates |
| **Code Size** | Minimal wrapper | Uses Scope as dependency |

## Tips

1. **Memory Usage**:
   - 40GB+ VRAM: Use `--quantization none` for best quality
   - 32GB VRAM: Use `--quantization fp8` to fit in memory
2. **Performance**: Each block generates ~11 frames. At 24fps, this is ~0.46s per block. Use `--num-blocks` to control video length:
   - `NUM_BLOCKS=9` (default) → ~99 frames → ~4.1 seconds
   - `NUM_BLOCKS=3` → ~33 frames → ~1.4 seconds
   - `NUM_BLOCKS=24` → ~264 frames → ~11 seconds
3. **Consistency**: Use the same `--seed` value for reproducible results
4. **Resolution**: Default 832x480 is optimized for performance. Higher resolutions require more VRAM.

## Architecture

This example uses:
- **Scope framework**: Production-ready video generation pipeline
- **Krea 14B checkpoint**: Real-time optimized video model
- **Wan 2.1 T2V 1.3B**: Base model components (VAE, text encoder)
- **Flash Attention 2**: Optimized attention mechanism

Models are automatically downloaded and cached to `~/.cache/cvlization/huggingface/`.

## Troubleshooting

**Blackwell GPU Compatibility (sm_120)**:
- **Status**: ✅ Working with PyTorch 2.9.1 + CUDA 12.8
- Successfully tested on NVIDIA RTX PRO 6000 (Blackwell architecture)
- If using older PyTorch versions, you may need to upgrade to 2.9.1+ for Blackwell support
- All integration issues have been resolved (flash-attention, model architecture, file paths, Blackwell support)

**Out of Memory (OOM)**:
- Use `--quantization fp8` for 32GB GPUs
- Reduce `--num-blocks` for shorter videos
- Lower resolution with `--width 640 --height 384`

**Model Download Issues**:
- Models automatically download to `~/.cache/cvlization/`
- Total download: ~45GB (base model + Krea checkpoint)
- Ensure sufficient disk space and stable internet

**Scope Installation Issues**:
- Scope is installed from GitHub during Docker build
- If build fails, check Scope repository is accessible
- Flash-attn is optional and may fail on some systems (non-critical)

## See Also

- [Scope Framework](https://github.com/daydreamlive/scope) - Production video generation framework
- [Krea AI](https://www.krea.ai/blog/krea-realtime-14b) - Official Krea documentation
- [krea-realtime](../krea_realtime/) - Alternative example using official SDK (currently blocked)

## License

This example follows the licenses of:
- Scope framework (check their repository)
- Krea model (Creative Commons Attribution-NonCommercial-ShareAlike 4.0)
