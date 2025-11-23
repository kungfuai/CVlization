# MapAnything - 3D Reconstruction

Inference example for MapAnything, a transformer model for 3D reconstruction tasks (SfM, MVS, depth estimation, etc.).

## Overview

MapAnything processes images to output 3D geometry and camera parameters. Supports single or multi-view input.

**Outputs:**
- 3D points
- Depth maps
- Camera intrinsics and poses
- Confidence scores

**Tasks supported:** Structure-from-Motion, Multi-View Stereo, monocular depth estimation, registration, depth completion (12+ total)

## Requirements

- GPU with 16GB+ VRAM (tested on RTX PRO 6000 Blackwell)
- Requires PyTorch 2.9.0+ for Blackwell GPU support
- 32GB RAM recommended
- ~20GB disk space

## Quick Start

### Build

```bash
./build.sh
```

### Run inference

```bash
# Place images in data/images/
mkdir -p data/images
# Copy your images here

# Run
./predict.sh

# Results saved to output/
```

### Custom paths

```bash
./predict.sh --images /path/to/images --output /path/to/output
```

## Output Files

- `camera_intrinsics.npy` - Camera intrinsic matrix (3x3)
- `summary.txt` - Output shapes and metadata

Note: The current implementation saves camera intrinsics. 3D points, depth maps, and poses are available in the predictions but need additional processing to save properly.

## Advanced Usage

### Use Apache-licensed model

```bash
./predict.sh --model facebook/map-anything-apache
```

### Precision options

```bash
./predict.sh --amp-dtype fp16  # Use FP16 instead of BF16
```

## Troubleshooting

### CUDA compatibility error

If you see "no kernel image is available for execution on the device", your PyTorch version doesn't support your GPU architecture. This example uses PyTorch 2.9.0 which supports Blackwell (sm_120).

### Out of memory

Reduce input image resolution or use CPU inference with `--device cpu`.

## Technical Details

- **Model:** facebook/map-anything (1.13GB DINOv2 backbone)
- **Base image:** pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
- **Input:** Single or multiple images (JPG, PNG)
- **Framework:** PyTorch with mixed precision (BF16/FP16)

Models are cached to `~/.cache/huggingface` and `~/.cache/torch`.

## Resources

- **Paper:** https://arxiv.org/abs/2501.05252
- **Repository:** https://github.com/facebookresearch/map-anything
- **Model Hub:** https://huggingface.co/facebook/map-anything
- **License:** CC-BY-NC 4.0 (research) / Apache 2.0 (commercial model available)

## Known Limitations

- Single image input works but multi-view input provides better reconstruction
- Current implementation only saves intrinsics; additional code needed to save 3D points/depth
- Requires modern GPU for practical inference speeds
