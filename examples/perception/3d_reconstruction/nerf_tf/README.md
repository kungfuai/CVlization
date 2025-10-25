# NeRF-TF: Neural Radiance Fields (Legacy)

⚠️ **LEGACY EXAMPLE** - This is an educational implementation for learning purposes only.

## Status

✅ **Fixed**: TensorFlow 2.16 migration issue resolved
⚠️ **Not recommended for production use**

## Recommended Modern Alternatives

For production 3D reconstruction and novel view synthesis, consider these modern alternatives:

1. **Instant-NGP** (NVIDIA, 2022)
   - 100-1000x faster training than original NeRF
   - Real-time rendering
   - CUDA-accelerated
   - https://github.com/NVlabs/instant-ngp

2. **3D Gaussian Splatting** (2023)
   - Real-time rendering quality
   - Faster training and inference
   - Better memory efficiency
   - https://github.com/graphdeco-inria/gaussian-splatting

3. **Nerfstudio** (2023)
   - Complete toolkit with viewer
   - Multiple NeRF variants
   - Production-ready framework
   - Active development and community
   - https://docs.nerf.studio/

## Why This is Legacy

- **Outdated**: NeRF (2020) is 3+ years old in a rapidly evolving field
- **Slow**: Training takes hours vs seconds for modern methods
- **Educational Only**: This is "Tiny NeRF" - a minimal demo, not production-ready
- **TensorFlow**: The 3D/NeRF community has standardized on PyTorch

## Usage (Educational)

```bash
# Build
./build.sh

# Train (requires dedicated GPU with ~12GB VRAM)
./train.sh
```

## Requirements

- GPU: 1x NVIDIA GPU with 12GB+ VRAM
- Note: Needs dedicated GPU memory (other processes may cause OOM)
- TensorFlow 2.16.1 with CUDA support

## What Was Fixed

The original TensorFlow migration error has been resolved:
- **Issue**: `ValueError: Cannot get result() since the metric has not yet been built`
- **Cause**: Metrics API incompatibility with `run_eagerly=True` in TF 2.x
- **Fix**: Removed metrics from compile, loss tracking still works