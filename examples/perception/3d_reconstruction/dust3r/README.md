# DUSt3R - Dense Unconstrained Stereo 3D Reconstruction

Inference example for DUSt3R, a feed-forward model for 3D reconstruction from uncalibrated images.

## Overview

DUSt3R performs dense 3D reconstruction from 2+ images without camera calibration or feature matching. Outputs dense point clouds, depth maps, and camera parameters.

**Outputs:**
- 3D scene (GLB format with cameras)
- Depth maps (PNG visualizations + NPY raw data)
- Confidence maps (per-pixel prediction confidence)
- Camera poses and focal lengths

**Inputs:** 2 or more images of the same scene

## Requirements

- GPU with 16GB+ VRAM
- Requires PyTorch 2.9.0+ with CUDA support (devel image for RoPE compilation)
- 32GB RAM recommended
- ~20GB disk space (model + dependencies)

## Quick Start

### Build

```bash
./build.sh
```

Build attempts CUDA kernel compilation for RoPE (Rotary Position Embeddings). Compilation may fail on some GPU architectures (e.g., Blackwell) but inference works with PyTorch fallback.

### Run inference

```bash
# Uses example images from hunyuanworld_mirror (2 images, mounted at runtime)
./predict.sh

# Results saved to outputs/
```

### Custom images

```bash
./predict.sh --input /path/to/images --output /path/to/output
```

The input directory is mounted into the container at runtime, so you can point to any directory on your host system.

## Output Files

```
outputs/
├── scene.glb                  # 3D scene with cameras (viewable in Blender, MeshLab)
├── depth_0000.png             # Depth visualization (view 1)
├── depth_0000.npy             # Raw depth values
├── depth_0001.png             # Depth visualization (view 2)
├── depth_0001.npy             # Raw depth values
├── confidence_0000.png        # Confidence map (view 1)
└── confidence_0001.png        # Confidence map (view 2)
```

## Advanced Usage

### Adjust reconstruction parameters

```bash
# Increase global alignment iterations (default: 300)
./predict.sh --niter 500

# Export as point cloud instead of mesh
./predict.sh --as-pointcloud

# Lower confidence threshold (more points, lower reliability)
./predict.sh --min-conf-thr 2.0

# Different scene graph (useful for many images)
./predict.sh --scene-graph swin
```

### Image size

```bash
# Use 224px model (faster, less accurate)
./predict.sh --image-size 224 --model-name naver/DUSt3R_ViTLarge_BaseDecoder_224_linear
```

## Technical Details

- **Model:** naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt
- **Base image:** pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel
- **Architecture:** ViT-Large encoder, ViT-Base decoder, DPT head
- **Input:** Images (any resolution, automatically resized)
- **Framework:** PyTorch with dense stereo matching

### How it works

1. **Pairwise matching:** All image pairs are processed to predict dense correspondences
2. **Global alignment:** Point clouds are aligned into a consistent 3D coordinate system
3. **Optimization:** Camera poses and 3D points are jointly optimized (300 iterations default)
4. **Export:** Final scene exported as GLB with cameras and point cloud/mesh

### Dependencies

- roma (relative rotation matching)
- trimesh (3D mesh processing)
- einops (tensor operations)
- gradio (not used in this example)

Models are cached to `~/.cache/huggingface` and `~/.cache/torch`.

## Troubleshooting

### CUDA compilation errors

RoPE CUDA kernels are optional. Compilation may fail on newer GPU architectures but inference works with PyTorch fallback (slightly slower).

### Out of memory

- Reduce number of input images
- Use `--image-size 224` for smaller model
- Process fewer pairs by changing scene graph type

### Poor reconstruction quality

- Ensure images overlap significantly (>50% shared content)
- Add more images for better constraints
- Increase `--niter` for longer optimization (default: 300)
- Try different `--scene-graph` types (complete, swin, oneref)

## Notes

Differences from traditional SfM pipelines (e.g., COLMAP):
- No explicit feature detection/matching step
- No camera calibration required
- Feed-forward inference (single pass)
- Trade-off: may be less accurate for large image sets

For dynamic scenes or video, see: [Easi3R](https://github.com/Inception3D/Easi3R)

## Resources

- **Repository:** https://github.com/naver/dust3r
- **Model Hub:** https://huggingface.co/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt
- **Paper:** https://arxiv.org/abs/2312.14132
- **License:** CC BY-NC-SA 4.0 (non-commercial use only)

## Known Limitations

- Requires 2+ images (single image needs duplication)
- VRAM usage scales with image resolution and count
- Global alignment can fail with very dissimilar viewpoints
- Non-commercial license restricts usage

## References

- Original Repository: https://github.com/naver/dust3r
- Paper: "DUSt3R: Geometric 3D Vision Made Easy" (CVPR 2024)
- Model Card: https://huggingface.co/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt
