# MonST3R - 4D Reconstruction from Dynamic Videos

Inference example for MonST3R, a feed-forward model for 4D reconstruction from videos with motion.

## Overview

MonST3R extends DUSt3R to handle dynamic scenes. Processes video/image sequences to produce time-varying 3D point clouds with dynamic/static segmentation.

**Outputs:**
- 4D scene (GLB format with time-varying geometry)
- Camera trajectory (TUM format)
- Per-frame depth maps (PNG + raw data)
- Confidence maps
- Dynamic/static segmentation masks
- Camera intrinsics

**Inputs:** Video file or image sequence with motion

## Requirements

- **GPU with 80GB+ VRAM** (95GB recommended for safety margin)
  - Measured: ~80GB peak for 30 frames with `--not-batchify` (default)
  - 65 frames will use proportionally more VRAM
  - `--batchify` flag uses significantly more memory (not recommended)
- PyTorch 2.9.0+ with CUDA support
- 32GB RAM recommended
- ~25GB disk space (model + dependencies + checkpoints)

## Quick Start

### Build

```bash
./build.sh
```

Build clones MonST3R repo with submodules (croco, viser) including demo data (lady-running, 65 frames). Checkpoints download lazily on first run.

### Run inference

```bash
# Uses demo video sequence (lady-running, 65 frames from MonST3R repo)
# Demo data is included in container during build - no download needed!
./predict.sh

# Results saved to outputs/
```

### Custom video/images

```bash
# Provide your own images or video
./predict.sh --input /path/to/video_or_images --output /path/to/output
```

## Output Files

```
outputs/
├── scene.glb                  # 4D scene (viewable in Blender, MeshLab)
├── pred_traj.txt              # Camera trajectory (TUM format)
├── pred_intrinsics.txt        # Camera intrinsics
├── depth_*.png                # Depth visualizations per frame
├── conf_*.png                 # Confidence maps per frame
├── dynamic_mask_*.png         # Dynamic/static segmentation
└── rgb_*.png                  # RGB images per frame
```

## Checkpoints (Lazy Download)

Checkpoints download automatically on first inference run:

1. **MonST3R model** (2.13GB) - Auto-downloaded from HuggingFace
2. **SAM2** (~900MB) - Downloaded from Meta (dynamic segmentation)
3. **RAFT** (~?MB) - Downloaded from Dropbox (optical flow)

Cached to `~/.cache/huggingface` and `/opt/monst3r/third_party/`.

## Advanced Usage

### Memory optimization

```bash
# Default: memory-efficient mode (--not-batchify is default)
./predict.sh

# Enable batchify for faster inference (requires more VRAM, may OOM)
./predict.sh --batchify

# Reduce number of frames to lower memory usage
./predict.sh --num-frames 32
```

### Adjust reconstruction parameters

```bash
# Increase global alignment iterations (default: 300)
./predict.sh --niter 500

# Lower confidence threshold (more points, lower reliability)
./predict.sh --min-conf-thr 2.0

# Limit number of frames for large videos
./predict.sh --num-frames 100

# Different scene graph (useful for many frames)
./predict.sh --scene-graph swin
```

### Image size

```bash
# Use 224px model (faster, less memory, less accurate)
./predict.sh --image-size 224 --model-name Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_224_linear
```

## Technical Details

- **Model:** Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt
- **Base image:** pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel
- **Architecture:** ViT-Large encoder, ViT-Base decoder, DPT head
- **Input:** Video or image sequence (any resolution, automatically resized)
- **Framework:** PyTorch with dense stereo matching + optical flow

### How it works

1. **Load frames:** Extract frames from video or load image sequence
2. **Pairwise matching:** Process all frame pairs for dense correspondences
3. **Optical flow:** RAFT estimates motion between frames
4. **Global alignment:** Point clouds aligned across time with motion constraints
5. **Segmentation:** SAM2 separates dynamic/static regions
6. **Optimization:** Joint optimization of geometry, poses, and motion (300 iterations default)
7. **Export:** 4D scene with time-varying geometry and per-frame outputs

### Dependencies

- dust3r (included in monst3r repo)
- croco (git submodule - base architecture)
- viser (git submodule - 4D visualization, not used in this example)
- SAM2 (third_party - dynamic segmentation)
- RAFT (third_party - optical flow)
- roma, trimesh, einops, gdown

Models and code cached to avoid re-download.

## Troubleshooting

### Out of memory

- Default mode already uses `--not-batchify` for memory efficiency
- Reduce `--num-frames` (e.g., `--num-frames 15` for ~40GB VRAM)
- Use `--image-size 224` for smaller model (lower quality)
- Lower `--batch-size` (default: 16)
- **Note:** VRAM scales with frame count (~80GB for 30 frames measured)

### Checkpoint download failures

Checkpoints download on first run. If download fails:
- Check internet connection
- SAM2: wget from Meta servers
- RAFT: wget from Dropbox
- Checkpoints cache to `/opt/monst3r/third_party/` inside container

### Poor reconstruction quality

- Ensure video has significant motion (camera or objects moving)
- Static scenes: use DUSt3R instead
- Increase `--niter` for longer optimization (default: 300)
- More frames generally improves results (but uses more VRAM)

## Notes

Differences from DUSt3R:
- Handles **dynamic scenes** (motion required)
- **Much higher VRAM** requirements (~80GB for 30 frames vs 8-12GB)
- **Additional outputs**: dynamic masks, temporal point clouds, time-varying geometry
- **Additional dependencies**: SAM2, RAFT
- **Longer inference:** ~5-10x slower than DUSt3R
- Trade-off: Significantly more complex, but handles motion

For static scenes, use DUSt3R instead.

## Resources

- **Repository:** https://github.com/junyi42/monst3r
- **Model Hub:** https://huggingface.co/Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt
- **Paper:** https://arxiv.org/abs/2410.03825
- **License:** CC BY-NC-SA 4.0 (non-commercial use only)
- **Project Page:** https://monst3r-project.github.io/

## Known Limitations

- Requires motion in scene (not for static images)
- **High VRAM usage:** ~80GB measured for 30 frames (with `--not-batchify` default)
- Longer inference time than DUSt3R (~5-10x)
- Non-commercial license restricts usage (CC BY-NC-SA 4.0)
- Checkpoint downloads on first run (~3GB total)

## References

- Original Repository: https://github.com/junyi42/monst3r
- Paper: "MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion" (ICLR 2025)
- Based on DUSt3R: https://github.com/naver/dust3r
