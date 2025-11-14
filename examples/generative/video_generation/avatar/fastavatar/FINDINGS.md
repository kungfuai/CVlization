# FastAvatar Novel View Quality Investigation

## Executive Summary

The current CVlization implementation uses FastAvatar's **feedforward-only mode** (`no_guidance`), which produces poor-quality novel views that don't match the input person's appearance. Investigation reveals this is an **inherent limitation** of feedforward mode, not a bug. High-quality results require implementing **test-time optimization** (`full_guidance` mode).

## Problem Description

When rendering novel views from feedforward-generated 3D Gaussian Splats:
- **Front view**: Reasonable reconstruction quality
- **Side views**: Dark, wrong colors, view-inconsistent appearance
- **Identity**: Doesn't match input person - looks averaged/generic

### Example: Sample 422

**Input image**: `images/sample_422.png` (frontal view, well-lit)

**Feedforward novel views** (in `results/sample_422_novel_views_corrected/`):
- View 004 (180°, back): Washed out, pale colors
- View 005 (225°): Wrong lighting, dark tones
- View 006 (270°, profile): Very dark, blue tint
- View 007 (315°): Dark, doesn't resemble input person

**Ground truth multi-view data** (in `data/sample_422_multiview/images/`):
- 16 high-quality views from all angles
- Consistent lighting and appearance
- Clear facial features matching the person

## Root Cause Analysis

### Investigation Process

1. **Verified SH coefficient handling in render_novel_views.py**
   - Fixed bug: Was only using `sh0` (DC component)
   - Now correctly using full SH: `torch.cat([sh0, shN], 1)` with `sh_degree=3`
   - Quality still poor after fix

2. **Inspected SH coefficient values in saved PLY**
   ```python
   # Base template (from decoder checkpoint):
   SH range: [-4.0, 4.0]  ✓ Normal, well-constrained

   # Feedforward output (sample_422/splats.ply):
   SH range: [-116.0, 125.0]  ✗ EXTREME, unconstrained!
   ```

3. **Analyzed conditioning MLP behavior**
   - The MLP outputs **unconstrained deltas** to SH coefficients
   - No clamping, normalization, or regularization during feedforward
   - Results in view-dependent color artifacts

### Technical Explanation

FastAvatar's architecture:
```
Input Image → Encoder → W vector → Decoder (base splats + MLP deltas) → Splats
                                                     ↓
                                          Unconstrained SH deltas
                                          (range: ±100+)
```

**Why this happens**:
- The conditioning MLP is trained with **photometric loss** during multi-view training
- During training, extreme SH values get penalized through rendering loss
- In **feedforward inference**, there's NO photometric feedback
- MLP can output arbitrary values → view-inconsistent colors

**Why original authors designed it this way**:
- Feedforward mode is meant as **initialization** for test-time optimization
- Full_guidance mode refines splats using multi-view images and FLAME params
- 400-800 iterations of optimization regularize the SH coefficients
- This is similar to NeRF/NeuS "per-scene optimization" paradigm

## Comparison: Feedforward vs Full-Guidance Modes

| Aspect | Feedforward (no_guidance) | Full-Guidance (test-time optimization) |
|--------|---------------------------|----------------------------------------|
| **Input required** | Single image | Multi-view images (8-16 views) + FLAME params |
| **Processing time** | ~2 seconds | 400-800 iterations (~5-10 minutes) |
| **W vector** | Fixed (from encoder) | Optimized during test-time |
| **MLP weights** | Fixed (pretrained) | Optionally fine-tuned |
| **SH regularization** | None | Photometric loss from multi-view |
| **Novel view quality** | Poor (this issue) | High (matches input person) |
| **Use case** | Quick preview/initialization | Final high-quality reconstruction |

## Evidence from Original Repository

### Multi-view Data Available

The original FastAvatar repo includes complete test datasets:
```
/tmp/FastAvatar/data/422/
├── images/               # 16 multi-view images (00000_00.png to 00000_15.png)
├── flame_param/          # FLAME parameters (00000.npz)
├── fg_masks/            # 16 foreground masks
├── transforms.json      # Camera poses (16 cameras)
├── points3d.ply        # COLMAP 3D reconstruction
└── sparse/             # COLMAP sparse reconstruction
```

**Sample 422** (same person we tested):
- 16 views: frontal, profiles, 3/4 views from multiple angles
- Resolution: 550x802
- Camera model: Pinhole with known intrinsics
- All views show **consistent, high-quality appearance**

### Full-Guidance Script

`/tmp/FastAvatar/scripts/inference_feedforward_full_guidance.py`:
- Loads multi-view images and camera poses
- Initializes with feedforward pass
- Optimizes for 400-800 epochs:
  - First half (0-400): Optimize W vector only
  - Second half (401-800): Optimize conditioning MLP
- Uses losses: L1 (0.6) + SSIM (0.3) + LPIPS (0.1)
- Renders novel views at checkpoints
- Saves optimized splats.ply

## Attempted Fixes and Why They Don't Work

### ✗ Fix 1: Clamp SH coefficients
```python
colors = torch.clamp(colors, -10, 10)
```
**Result**: Reduces artifacts but still wrong colors - clamping arbitrary wrong values doesn't make them correct.

### ✗ Fix 2: Apply sigmoid activation
```python
colors = torch.sigmoid(colors)
```
**Result**: Maps to [0,1] but loses view-dependent effects - SH coefficients should be in [-∞, +∞] range.

### ✗ Fix 3: Normalize per-Gaussian
```python
colors = F.normalize(colors, dim=-1)
```
**Result**: Destroys color information - normalization is not appropriate for SH coefficients.

### ✓ Correct Solution: Test-time optimization
Implement full_guidance mode with multi-view supervision to regularize SH coefficients through photometric loss.

## Recommendations

### Option 1: Document Limitations (Quick)
- Add warning to README about feedforward-only quality
- Document that novel views are low quality
- Recommend users implement test-time optimization for production use
- Estimated effort: 30 minutes

### Option 2: Implement Simple SH Regularization (Medium)
- Add L2 regularization loss on SH coefficients during feedforward
- Penalize deviation from base template
- May improve quality slightly but won't match test-time optimization
- Estimated effort: 2-4 hours

### Option 3: Implement Full Test-Time Optimization (Recommended)
- Port `inference_feedforward_full_guidance.py` to CVlization
- Support multi-view input format
- Add optimization loop with photometric losses
- This is the only way to achieve paper-quality results
- Estimated effort: 1-2 days

### Option 4: Hybrid Mode (Advanced)
- Use feedforward for instant preview
- Automatically trigger optimization if multi-view data available
- Best user experience
- Estimated effort: 2-3 days

## Files Modified

### scripts/render_novel_views.py
**Fixed bugs**:
- Line 117-121: Now using full SH coefficients instead of DC-only
- Line 136-137: Added `sh_degree=3` parameter

**Current status**: Rendering code is correct, but input splats have extreme SH values from feedforward mode.

## Test Results

### Sample: Lenna (results/lenna_novel_views_corrected/)
- Input: Famous test image
- Feedforward splats: SH range [-89, 102]
- Novel views: Generic face, wrong hair color

### Sample: sample_422 (results/sample_422_novel_views_corrected/)
- Input: Multi-view dataset subject #422, frontal image
- Feedforward splats: SH range [-116, 125]
- Novel views: Very poor quality, dark, blue tint in profiles
- **Ground truth available**: 16 high-quality multi-view images in `data/sample_422_multiview/`

## Conclusion

The poor novel view quality is **not a bug** but an **inherent limitation** of feedforward-only inference in FastAvatar. The model was designed with test-time optimization as the intended path to high-quality results. To achieve results matching the FastAvatar paper:

1. Must implement test-time optimization with multi-view supervision
2. Feedforward mode alone is insufficient for production use
3. Original authors provide complete multi-view datasets for 6 test subjects
4. Implementation would follow NeRF/NeuS paradigm: per-subject optimization

**Current CVlization implementation**: Working as designed, but limited to preview quality.

**Next steps**: Decide between documenting limitations vs implementing full test-time optimization.

---

## Update: Full-Guidance Optimization Running

**Status**: Currently running test-time optimization on sample 422

- **Mode**: Full_guidance with 51 epochs per view (quick test)
- **Progress**: Optimizing 16 multi-view images sequentially
- **Loss reduction**: Significant improvement per view (e.g., View 1: 0.1483 → 0.0981)
- **Output**: `results/sample_422_full_guidance/`

This will demonstrate the quality improvement from test-time optimization compared to feedforward-only mode.
