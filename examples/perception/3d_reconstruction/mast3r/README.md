# MASt3R - Image Matching + Sparse 3D Reconstruction

Inference example for MASt3R: Grounding Image Matching in 3D with dense correspondences and metric alignment.

## Overview

MASt3R builds on DUSt3R to provide robust image matching and sparse 3D reconstruction from uncalibrated images. Uses dense correspondence matching with metric constraints for accurate scene geometry.

**Outputs:**
- 3D scene (GLB format with geometry and cameras)
- Per-frame depth maps (PNG + raw data)
- Confidence maps
- Camera poses (implicit in alignment)
- Dense correspondences

**Inputs:** Multiple uncalibrated images of the same scene

## Requirements

- **GPU with 6-8GB VRAM** (8GB recommended)
  - Measured: ~5GB peak for 7 images
  - More images = more VRAM usage
  - Similar efficiency to DUSt3R
- PyTorch 2.9.0+ with CUDA support
- 16GB RAM recommended
- ~15GB disk space (model + dependencies)

## Quick Start

### Build

```bash
./build.sh
```

Build clones MASt3R repo with submodules (including DUSt3R) and demo data. Model downloads lazily on first run.

### Run inference

```bash
# Uses demo image sequence (NLE_tower, 7 images from MASt3R repo)
# Demo data is included in container during build - no download needed!
./predict.sh

# Results saved to outputs/
```

### Custom images

```bash
# Provide your own images
./predict.sh --input /path/to/images --output /path/to/output
```

## Output Files

```
outputs/
├── scene.glb                  # 3D scene (viewable in Blender, MeshLab)
├── depth_*.png                # Depth visualizations per view
├── depth_*.npy                # Raw depth maps per view
└── confidence_*.png           # Confidence maps per view
```

## Model Download (Lazy)

Model downloads automatically on first inference run:

- **MASt3R model** (~1.6GB) - Auto-downloaded from HuggingFace: `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`

Cached to `~/.cache/huggingface`.

## Advanced Usage

### Adjust reconstruction parameters

```bash
# Increase global alignment iterations (default: 300)
./predict.sh --niter 500

# Lower confidence threshold (more points, lower reliability)
./predict.sh --min-conf-thr 2.0

# Different scene graph (useful for many images)
./predict.sh --scene-graph swin
```

### Image size

```bash
# Use 224px model (faster, less memory, less accurate)
./predict.sh --image-size 224 --model-name naver/MASt3R_ViTLarge_BaseDecoder_224_catmlpdpt_metric
```

### Export options

```bash
# Export as point cloud instead of mesh
./predict.sh --as-pointcloud
```

## Technical Details

- **Model:** naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
- **Base image:** pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel
- **Architecture:** ViT-Large encoder, ViT-Base decoder, CatMLP+DPT head with metric alignment
- **Input:** Multiple images (any resolution, automatically resized)
- **Framework:** PyTorch with dense stereo matching + global optimization

### How it works

1. **Load images:** Load image sequence from directory
2. **Pairwise matching:** MASt3R processes all image pairs for dense correspondences
3. **Metric alignment:** Enforces metric constraints during matching (vs. DUSt3R's relative depth)
4. **Global alignment:** Point clouds aligned across views (300 iterations default)
5. **Optimization:** Joint optimization of geometry and camera poses
6. **Export:** 3D scene with geometry, cameras, and per-frame outputs

### Key Differences from DUSt3R

- **Metric alignment:** MASt3R uses metric constraints for better absolute scale
- **Better matching:** Improved correspondence matching for visual localization
- **Same requirements:** Similar VRAM and speed to DUSt3R
- **Compatible:** Uses DUSt3R infrastructure and can load DUSt3R models

### Dependencies

- dust3r (included as git submodule)
- croco (DUSt3R's base architecture)
- roma, trimesh, einops
- scikit-learn (for sparse reconstruction)

Models and code cached to avoid re-download.

## Troubleshooting

### Out of memory

- Reduce number of images
- Use `--image-size 224` for smaller model
- Lower `--batch-size` (default: 1)
- VRAM scales with image count (~5GB for 7 images measured)

### Model download failures

Model downloads on first run. If download fails:
- Check internet connection
- Verify HuggingFace access
- Model caches to `~/.cache/huggingface` inside container

### Poor reconstruction quality

- Ensure images have sufficient overlap (>50% recommended)
- More images generally improve results (diminishing returns after ~10-20)
- Increase `--niter` for longer optimization (default: 300)
- Use `--scene-graph complete` for exhaustive matching (default)

## Comparison with DUSt3R

| Feature | DUSt3R | MASt3R |
|---------|--------|--------|
| Architecture | ViT-L encoder + decoder | ViT-L encoder + decoder |
| Output | Relative depth | Metric depth |
| Matching | Dense correspondences | Dense + metric constraints |
| VRAM | 6-8GB | 6-8GB |
| Speed | ~baseline | Similar to DUSt3R |
| Best for | General 3D reconstruction | Visual localization, mapping |

**When to use MASt3R over DUSt3R:**
- Visual localization tasks
- When metric scale matters
- When you need better correspondence matching

**When to use DUSt3R instead:**
- When relative depth is sufficient
- Simpler use cases

## Notes

- For static scenes only (for dynamic scenes, use MonST3R)
- Non-commercial license restricts usage (CC BY-NC-SA 4.0)
- Built on top of DUSt3R (same team from NAVER Labs)
- First presented at ECCV 2024

## Resources

- **Repository:** https://github.com/naver/maSt3R
- **Model Hub:** https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
- **Paper:** https://arxiv.org/abs/2406.09756 (ECCV 2024)
- **License:** CC BY-NC-SA 4.0 (non-commercial use only)
- **Project Page:** https://europe.naverlabs.com/research/mast3r

## Known Limitations

- Requires static scenes (not for videos with motion)
- VRAM usage scales with image count (~5GB for 7 images)
- Non-commercial license restricts usage (CC BY-NC-SA 4.0)
- Model downloads on first run (~1.6GB)

## References

- Original Repository: https://github.com/naver/maSt3R
- Paper: "Grounding Image Matching in 3D with MASt3R" (ECCV 2024)
- Based on DUSt3R: https://github.com/naver/dust3r
