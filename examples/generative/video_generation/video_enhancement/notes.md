# Video Artifact Removal

A deep learning model for removing visual artifacts from videos, optimized for Apple M4 inference.

## Architecture

### Current (v1)
- **2D U-Net with NAFNet-style blocks** - Efficient and well-supported on Apple Silicon
- **Optional temporal attention** - For video consistency without 3D convolutions
- **~5-10M parameters** - Suitable for real-time inference on M4

### Planned Improvements (v2)

Based on recent research (ProPainter ICCV 2023, VideoPainter SIGGRAPH 2024, DFCL 2025):

```
                              ┌─────────────────┐
                              │   Mask Head     │ ← auxiliary supervision
                              │   (learned)     │
                              └────────┬────────┘
                                       │ soft mask
                                       ▼
video → Encoder → ┌──────────────────────────────────┐
                  │  Mask-Guided Sparse Attention    │
                  │  (attend only where mask > τ)    │
                  └──────────────────────────────────┘
                                       │
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
            Flow Completion    Feature Propagation    Inpaint Head
                  │                    │                    │
                  └────────────────────┴────────────────────┘
                                       │
                                   Decoder → clean video
```

**Key insights from recent work:**

1. **Mask-guided sparse attention (ProPainter)**: The mask isn't just "where to fix" — it's for computational efficiency. Watermarks typically occupy <15% of frame, so sparse attention gives ~6-7x speedup by skipping non-watermark tokens.

2. **Dual-pathway encoder (DFCL)**: RGB + gradient (Sobel) pathways. Gradients help detect watermark edges and high-frequency artifacts.

3. **Flow completion**: Critical for temporal consistency. Prevents flickering by propagating information along motion trajectories.

4. **Soft masks + joint training**: Hard masking causes boundary artifacts. Soft masks with decaying supervision let network learn to fix things regardless of mask accuracy.

**Proposed changes:**

| Current | Proposed | Reason |
|---------|----------|--------|
| Full temporal attention | Mask-guided sparse attention | Efficiency: skip computation on non-watermark regions |
| Single encoder | Dual-pathway (RGB + gradient) | Gradients help detect watermark edges |
| No flow | Flow completion network | Critical for video temporal consistency |
| Optional mask head | Mask + flow + inpaint heads | Decouple tasks for better specialization |
| Hard mask (if used) | Soft mask for token selection | Avoid boundary artifacts |

**Training strategy:**
1. **Stage 1 (warm-up)**: Train encoder + mask head with mask supervision only
2. **Stage 2 (main)**: Joint training with all losses, mask loss weight decays
3. **Stage 3 (fine-tune)**: Remove mask supervision, let network decide

**Proposed loss function:**
```python
loss = (
    1.0 * charbonnier(pred, gt) +
    0.1 * lpips(pred, gt) +
    0.5 * temporal_consistency(pred, gt, flow) +
    0.3 * bce(pred_mask, gt_mask) +  # Mask supervision (if available)
    0.1 * edge_loss(pred, gt)  # From gradient pathway
)
```

**References:**
- ProPainter (ICCV 2023): Mask-guided sparse video transformer
- VideoPainter (SIGGRAPH 2024): Dual-branch for background preservation + foreground generation
- DFCL (Neural Networks 2025): Dual-pathway RGB + gradient for watermark removal
- SplitNet (AAAI 2021): Split detection/removal, then refine with RefineNet

## Project Structure

```
video_enhancement/
├── config.py           # Configuration dataclasses
├── model.py            # Neural network architecture
├── losses.py           # Loss functions (pixel, perceptual, temporal, FFT)
├── visual_artifacts.py # Synthetic artifact generation
├── dataset.py          # Dataset and data loading
├── train.py            # Training script
├── predict.py          # Inference script (optimized for M4)
└── requirements.txt    # Dependencies
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train with dummy data (for testing)

```bash
python train.py --dummy --epochs 10 --batch-size 4
```

### 3. Train with real data

Organize your data:
```
data/
├── train/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── val/
    ├── video1.mp4
    └── ...
```

Then train:
```bash
python train.py --data ./data --epochs 100
```

### 4. Inference

```bash
# Single image
python predict.py -i degraded.png -o clean.png -c checkpoints/best.pt

# Video
python predict.py -i degraded.mp4 -o clean.mp4 -c checkpoints/best.pt

# Benchmark
python predict.py -c checkpoints/best.pt --benchmark
```

## Training Options

```bash
python train.py \
    --data ./data \           # Path to data directory
    --epochs 100 \            # Number of epochs
    --batch-size 4 \          # Batch size
    --lr 1e-4 \               # Learning rate
    --no-temporal \           # Disable temporal attention
    --no-lpips \              # Use VGG instead of LPIPS
    --device mps \            # Force specific device
    --resume checkpoint.pt \  # Resume from checkpoint
    --residual \              # Use residual learning
    --artifacts "corner_logo,gaussian_noise"  # Specify artifact types
```

## Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| Charbonnier (L1-like) | 1.0 | Pixel reconstruction |
| LPIPS/VGG Perceptual | 0.1 | High-level feature matching |
| Temporal consistency | 0.5 | Reduce flickering |
| FFT | 0.05 | Preserve textures |

## Performance (Expected)

| Resolution | M4 Mac Mini | A10 GPU |
|------------|-------------|---------|
| 256x256 | ~50 FPS | ~100 FPS |
| 512x512 | ~20 FPS | ~60 FPS |
| 720p | ~10 FPS | ~30 FPS |

## Supported Artifact Types

### Overlay Artifacts (use masks)
- `corner_logo` - Static logo in corner
- `text_overlay` - Text overlays
- `tiled_pattern` - Repeating patterns
- `moving_logo` - Animated/moving logos
- `channel_logo` - TV channel style logos
- `diagonal_text` - Diagonal text overlays

### Degradation Artifacts (modify pixels)
- `jpeg_compression` - JPEG compression artifacts
- `video_compression` - Video codec artifacts (blocking, banding)
- `gaussian_noise` - Gaussian noise
- `salt_pepper_noise` - Salt and pepper noise
- `film_grain` - Film grain effect
- `color_banding` - Reduced bit depth banding
- `blur` - Blur artifacts

## Tips for Best Results

1. **Start with 256x256** for faster iteration, then scale up
2. **Use diverse artifacts** during training for generalization
3. **Add real degraded data** if available (with clean pairs)
4. **Temporal loss is crucial** for video quality
5. **For M4 deployment**, consider exporting to Core ML

## Export to Core ML (M4 Optimization)

```python
import coremltools as ct
import torch

# Load trained model
model = load_model("checkpoints/best.pt", torch.device("cpu"))
model.eval()

# Trace
example = torch.randn(1, 3, 256, 256)
traced = torch.jit.trace(model, example)

# Convert
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=example.shape)],
    minimum_deployment_target=ct.target.macOS14,
)

mlmodel.save("ArtifactRemoval.mlpackage")
```

## Known Limitations

- ConvTranspose3d not supported on MPS (that's why we use 2D architecture)
- Very long videos may need chunked processing
- Moving/animated artifacts are harder than static ones

## Open Questions (v2 Architecture)

1. **Mask supervision source**: Do we have GT masks, or use self-supervised (degraded - clean difference)?

2. **Scope of flow completion**: Full ProPainter-style is substantial. Options:
   - (a) Mask-guided sparse attention only (moderate change)
   - (b) Add dual-pathway encoder (smaller change)
   - (c) Full overhaul with flow completion (major change)

3. **Artifact type handling**: Watermarks benefit from masks, but compression/noise artifacts are global. Keep both paths?
   - Overlay artifacts → mask-guided sparse attention
   - Degradation artifacts → full attention (or different branch)

4. **Efficiency vs accuracy tradeoff**: Sparse attention threshold τ selection. Too high = miss artifacts, too low = lose efficiency gains.

## Implementation Roadmap

> **Note**: Keep the existing `ArtifactRemovalNet` (v1) intact. New architecture should be a separate class (e.g., `ArtifactRemovalNetV2`). Add `--net {v1,v2}` flag to `train.py` and `predict.py` to select which network to use.

### Phase 1: Quick wins
- [ ] Enable edge loss (`w_edge=0.1`) — already implemented in losses.py
- [ ] Add gradient (Sobel) input channel to encoder
- [ ] Use mask head output to weight attention (soft gating)

### Phase 2: Sparse attention
- [ ] Implement `MaskGuidedSparseAttention` module
- [ ] Benchmark efficiency gains vs current `TemporalAttention`
- [ ] Tune sparsity threshold τ

### Phase 3: Flow completion
- [ ] Add lightweight flow estimation (RAFT-small or PWC-Net)
- [ ] Implement flow-guided feature propagation
- [ ] Add flow consistency loss

### Phase 4: Multi-stage training
- [ ] Implement staged training with mask loss decay
- [ ] Evaluate on real watermarked videos
- [ ] Compare v1 vs v2 on PSNR/SSIM/LPIPS

## License

MIT
