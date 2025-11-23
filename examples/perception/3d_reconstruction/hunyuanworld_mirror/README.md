# HunyuanWorld-Mirror - 3D Reconstruction

Inference example for HunyuanWorld-Mirror, a feed-forward model for 3D geometric prediction from images.

## Overview

HunyuanWorld-Mirror processes images to generate multiple 3D representations in a single forward pass.

**Outputs:**
- Point clouds (.ply)
- Depth maps (.png, .npy)
- Surface normals (.png)
- 3D Gaussians (.ply)
- Camera parameters
- Optional: COLMAP reconstruction, rendered videos

**Inputs:** Images or video (single or multi-view)

## Requirements

- GPU with 20GB+ VRAM
- Requires PyTorch 2.9.0+ for Blackwell GPU support
- 32GB RAM recommended
- ~25GB disk space

## Quick Start

### Build

```bash
./build.sh
```

### Run inference

```bash
# Uses included Desk example images
./predict.sh

# Results saved to outputs/
```

### Custom images

```bash
./predict.sh --input /path/to/images --output /path/to/output
```

## Output Files

```
outputs/
├── Desk/                           # Scene name from input
│   ├── sparse/                     # (if --save-colmap enabled)
│   │   ├── points3D.bin
│   │   ├── cameras.bin
│   │   └── images.bin
│   ├── points3D.ply                # Point cloud
│   ├── depth_0000.png              # Depth map (per view)
│   ├── normal_0000.png             # Surface normals (per view)
│   └── gaussians.ply               # 3D Gaussians (if gsplat installed)
```

## Advanced Usage

### Enable optional outputs

```bash
# Save COLMAP reconstruction
./predict.sh --save-colmap

# Save rendered interpolation video
./predict.sh --save-rendered

# Apply sky segmentation filtering
./predict.sh --apply-sky-mask
```

### Control image size

```bash
# Process at different resolution (default: 518px)
./predict.sh --target-size 768
```

### Video input

```bash
# Process video file
./predict.sh --input video.mp4
```

## Technical Details

- **Model:** tencent/HunyuanWorld-Mirror
- **Base image:** pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
- **Input:** Images (.jpg, .png, .webp) or video (.mp4, .avi, .mov, .webm, .gif)
- **Framework:** PyTorch with multi-modal prior prompting

### Dependencies

- pycolmap (COLMAP Python bindings)
- gsplat (3D Gaussian Splatting - best-effort install)
- open3d (point cloud processing)
- onnxruntime (sky segmentation)

Models are cached to `~/.cache/huggingface` and `~/.cache/torch`.

## Troubleshooting

### gsplat installation warning

The build may show a warning about gsplat installation. This is expected - gsplat wheels may not exist for all PyTorch/CUDA combinations. The example will work without it, but 3D Gaussian output will be unavailable.

### CUDA compatibility error

If you see "no kernel image available", your PyTorch version doesn't support your GPU architecture. This example uses PyTorch 2.9.0 which supports Blackwell (sm_120).

### Out of memory

Reduce image count, lower resolution with `--target-size 384`, or process fewer views at once.

## Resources

- **Repository:** https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror
- **Model Hub:** https://huggingface.co/tencent/HunyuanWorld-Mirror
- **License:** Apache 2.0

## Known Limitations

- gsplat may not install on all platforms (Gaussian output unavailable)
- COLMAP reconstruction can be slow for many views
- Video processing extracts frames at specified FPS (increases memory usage)
- Sky segmentation requires additional ONNX model download (~10MB)
