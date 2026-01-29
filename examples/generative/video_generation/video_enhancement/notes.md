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

**Mask Guidance Variants:**

Different strategies for how predicted mask can guide the inpainting process:

### 1. Feature Modulation (current implementation)
```python
# Modulate features where mask indicates artifacts
feat = feat * (1 + scale * mask) + bias * mask
```
- Learned scale/bias per channel
- Additive + multiplicative adjustment
- Applied after final decoder stage

### 2. Concatenation (simplest baseline)
```python
# Concat mask as extra input channel to decoder
feat = torch.cat([feat, mask], dim=1)
feat = conv(feat)  # network learns how to use mask
```
- Let network learn how to use mask
- No architectural assumptions
- Can apply at any resolution

### 3. Attention Gating
```python
# Use mask to gate attention weights - focus on artifact regions
attn_weights = softmax(Q @ K.T)
attn_weights = attn_weights * (1 + alpha * mask)  # boost artifact regions
out = attn_weights @ V
```
- Attention focuses more on artifact regions
- Helps propagate clean context into artifacts
- Good for temporal attention

### 4. Skip Connection Gating
```python
# Gate skip connections based on mask
# Hypothesis: encoder features in artifact regions are "corrupted", suppress them
skip_gated = skip * (1 - mask) + skip * mask * gate_weight
feat = decoder_block(feat, skip_gated)
```
- Suppress encoder features in artifact regions
- Force decoder to hallucinate from context, not corrupted input
- `gate_weight` can be learned or fixed (e.g., 0.1)

### 5. Dual-Stream Decoder
```python
# Separate processing for clean vs artifact regions
clean_feat = feat * (1 - mask)
artifact_feat = feat * mask

clean_out = preservation_branch(clean_feat)   # identity-like, lightweight
artifact_out = inpainting_branch(artifact_feat)  # generative, heavier

out = clean_out + artifact_out
```
- Specialized branches for different tasks
- Preservation branch can be very lightweight (just pass-through)
- Similar to VideoPainter's dual-branch philosophy

### 6. Residual Weighting
```python
# Only predict/apply residual in mask regions
residual = residual_net(feat)
out = input + residual * mask  # residual only applied where mask > 0
```
- Explicit: no change outside mask
- Like explicit composite but at residual level
- Gradients only flow through mask regions

### 7. Cross-Attention (Clean → Artifact)
```python
# Query artifact regions, Key/Value from clean regions
Q = project_q(feat * mask)           # artifact region queries
K = project_k(feat * (1 - mask))     # clean region keys
V = project_v(feat * (1 - mask))     # clean region values

# Artifact regions "borrow" features from clean regions
borrowed_features = softmax(Q @ K.T) @ V
feat_artifact = feat * mask + borrowed_features
```
- Explicitly borrow texture/patterns from clean regions
- Good for pattern/texture completion
- Can be expensive (full attention matrix)

### Comparison

| Variant | Complexity | Interpretability | Best For |
|---------|------------|------------------|----------|
| Modulation | Low | Medium | General purpose |
| Concatenation | Lowest | Low | Simple baseline |
| Attention Gating | Medium | Medium | Large artifacts |
| Skip Gating | Medium | High | Suppressing corrupted features |
| Dual-Stream | High | High | Different artifact types |
| Residual Weighting | Low | High | Overlay artifacts |
| Cross-Attention | High | High | Texture borrowing |

**Recommendations:**
1. Start with **Modulation** (current) or **Concatenation** (simpler)
2. Try **Skip Gating** if encoder features seem corrupted
3. Use **Cross-Attention** for texture-heavy inpainting

**References:**
- ProPainter (ICCV 2023): Mask-guided sparse video transformer
- VideoPainter (SIGGRAPH 2024): Dual-branch for background preservation + foreground generation
- DFCL (Neural Networks 2025): Dual-pathway RGB + gradient for watermark removal
- SplitNet (AAAI 2021): Split detection/removal, then refine with RefineNet

## Literature Review (2023-2025)

### Video Inpainting

#### VideoPainter (Bian et al., TOG 2024)
**Dual-branch Diffusion Transformer** - State-of-the-art method combining lightweight CNN context encoder with pre-trained video diffusion Transformer (DiT).

| Aspect | Details |
|--------|---------|
| Architecture | Dual-branch: CNN encoder (~6% of backbone) + DiT backbone |
| Pretraining | Latent diffusion model on large video/image data (no ImageNet) |
| VAE | Yes - frames encoded to latent space |
| Inference | Multi-step diffusion sampling (iterative) |
| Auxiliary | No mask/flow prediction - mask is input |
| Training Data | VPData: ~390k clips (~867 hours) - largest video inpainting dataset |
| Key Innovation | "Inpainting region ID resampling" for long video identity consistency |

#### AVID (Zhang et al., CVPR 2024)
**Any-Length Video Inpainting with Diffusion** - Builds on Stable Diffusion inpainting with temporal modules.

| Aspect | Details |
|--------|---------|
| Architecture | Latent diffusion U-Net + temporal self-attention (AnimateDiff-style) |
| Pretraining | Stable Diffusion inpainting backbone (LAION, not ImageNet) |
| VAE | Yes - Stable Diffusion VAE |
| Inference | Temporal Multi-Diffusion with middle-frame attention guide |
| Auxiliary | Structure guidance module (HED edge maps) - adjustable per task |
| Training | Two-stage: temporal modules, then structure module |
| Open Source | Yes - GitHub |

#### CoCoCo (Zi et al., AAAI 2024)
**Consistency, Controllability, Compatibility** - Text-guided diffusion with enhanced temporal attention.

| Aspect | Details |
|--------|---------|
| Architecture | Stable Diffusion + motion capture module (3 attention types) |
| Key Features | Damped global temporal attention, instance-aware mask selection |
| Compatibility | Plug-and-play with LoRA/DreamBooth models |
| Open Source | Yes - Video-Inpaint-Anything |

#### OmniPainter (ICLR 2026 submission)
**Flow-Guided Latent Diffusion** - Adaptive global-local guidance for temporal consistency.

| Aspect | Details |
|--------|---------|
| Key Innovation | Flow-Guided Ternary Control + Adaptive Global-Local Guidance |
| Strategy | Early steps: global structure; Late steps: local details |
| Flow | Uses off-the-shelf optical flow (not learned) |

#### FGDVI (Gu et al., 2024)
**Flow-Guided Diffusion (Training-Free)** - Inference-time technique for any diffusion model.

| Aspect | Details |
|--------|---------|
| Approach | Warp latents via optical flow between frames |
| Advantage | No training needed - works with existing models |
| Open Source | Yes - GitHub |

### Video Artifact Removal (Compression Enhancement)

#### STFF (Wang et al., IEEE TBC 2025)
**Spatio-Temporal & Frequency Fusion** - State-of-the-art for compression artifact removal.

| Aspect | Details |
|--------|---------|
| Architecture | CNN-based, 3 stages |
| Stage 1 | Feature Extraction & Alignment (FEA) - RNN-based (SRU) for spatio-temporal features |
| Stage 2 | Bidirectional High-Frequency Enhanced Propagation (BHFEP) - forward/backward with HCAB |
| Stage 3 | Residual High-Frequency Refinement (RHFR) - recover textures/edges |
| Pretraining | None - trained from scratch (artifact patterns are compression-specific) |
| VAE | No - direct regression model |
| Inference | One-pass (real-time capable) |
| Auxiliary | Implicitly targets high-frequency residuals via architecture |
| Open Source | Yes - GitHub |

### Key Takeaways

#### Video Inpainting Trends (2023-2025)
- **Shift to diffusion models** - From CNN-based to diffusion Transformers
- **Leverage pretrained models** - Use large video/image diffusion backbones, add temporal modules
- **VAE latent space** - Standard practice for efficiency
- **Optical flow guidance** - Critical for temporal consistency (external or learned)
- **High compute** - Large datasets (100k+ clips), multi-step inference

#### Video Artifact Removal Trends
- **CNN/Transformer-based** - Not diffusion (deterministic task)
- **No ImageNet backbone** - Train from scratch on compressed video pairs
- **Multi-frame aggregation** - Use neighboring frames for restoration
- **Frequency-domain focus** - Target high-frequency details lost to compression
- **One-pass inference** - Suitable for real-time/streaming

#### Implications for Our Model

| SOTA Approach | Our Current Approach | Gap |
|---------------|---------------------|-----|
| Pretrained diffusion backbone | Train from scratch | Could use pretrained encoder |
| VAE latent space | Direct pixel space | Simpler but less efficient |
| Multi-step diffusion | One-pass forward | Faster but less powerful |
| Optical flow guidance | Temporal attention only | Could add flow |
| Large datasets (100k+ clips) | Vimeo-90K + synthetic | Adequate for overlay artifacts |

**Recommendation**: For watermark/overlay removal (our focus), one-pass CNN approach is appropriate. Key improvements:
1. Add optical flow for temporal consistency
2. Consider pretrained encoder (but increases model size)
3. Frequency-domain losses (already have FFT loss)

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
- Very long videos may need chunked processing (see Long Video Inference below)
- Moving/animated artifacts are harder than static ones

## Long Video Inference

For videos longer than the training clip length, three main strategies exist:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LONG VIDEO INFERENCE OPTIONS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. SLIDING WINDOW (Simple, but boundary artifacts)                          │
│     ┌────┐ ┌────┐ ┌────┐ ┌────┐                                             │
│     │ W1 │→│ W2 │→│ W3 │→│ W4 │→ ...                                        │
│     └────┘ └────┘ └────┘ └────┘                                             │
│        ↑overlap↑                                                             │
│                                                                              │
│  2. RECURRENT/STREAMING (Online, memory efficient)                           │
│     frame₁ → [net + h₀] → frame₁' + h₁                                      │
│     frame₂ → [net + h₁] → frame₂' + h₂                                      │
│     frame₃ → [net + h₂] → frame₃' + h₃  ...                                 │
│                                                                              │
│  3. CLIP-RECURRENT (Best of both - RVRT/VideoPainter style)                  │
│     ┌──────────┐     ┌──────────┐     ┌──────────┐                          │
│     │ Clip 1   │────→│ Clip 2   │────→│ Clip 3   │→ ...                     │
│     │ (parallel│ h₁  │ (parallel│ h₂  │ (parallel│                          │
│     │  within) │     │  within) │     │  within) │                          │
│     └──────────┘     └──────────┘     └──────────┘                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Strategy Comparison

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **Sliding Window** | Simple, parallel | Boundary flicker, no long-range | Short clips (<5s) |
| **Recurrent** | Memory efficient, online | Error accumulates, quality degrades | Real-time streaming |
| **Clip-Recurrent** | Quality + consistency | Slightly more complex | Production (recommended) |
| **Bidirectional** | Best quality | 2x memory, 2x compute, offline only | Highest quality offline |

### Recommended: Clip-Recurrent with Hidden State

For watermark removal, **clip-recurrent** is ideal because:
- Watermarks are temporally stable → hidden state helps track them
- Watermarks don't need "future" context → unidirectional is fine
- M4 memory constraints → clip-based fits better than full bidirectional

```python
class LongVideoInference:
    """
    Process video in clips, passing hidden state between clips.
    Parallel within clip, sequential across clips.
    """
    def __init__(self, model, clip_length=16, overlap=4):
        self.model = model
        self.clip_length = clip_length
        self.overlap = overlap
        self.stride = clip_length - overlap

    def __call__(self, video: Tensor) -> Tensor:
        T = video.shape[0]
        outputs = []
        hidden = None

        for start in range(0, T, self.stride):
            end = min(start + self.clip_length, T)
            clip = video[start:end]

            # Process clip (parallel across frames within clip)
            out_clip, hidden = self.model(clip, hidden)

            # Only append non-overlapping frames
            if outputs and self.overlap > 0:
                out_clip = out_clip[self.overlap:]

            outputs.append(out_clip)
            hidden = self._detach_hidden(hidden)  # Prevent memory buildup

        return torch.cat(outputs, dim=0)[:T]
```

### What Goes in Hidden State?

For watermark removal:
1. **Feature maps** from last N frames (for temporal attention)
2. **Optical flow** from previous clip (for propagation)
3. **Predicted masks** from previous clip (temporal consistency)
4. **Texture bank** (optional): reference clean textures found so far

### Training Considerations

Training with short clips causes domain gap with long video inference (hidden states differ). Solution: train with random hidden state initialization.

```python
def train_step(model, long_video, clip_length=16):
    T = long_video.shape[0]
    start = random.randint(0, T - clip_length)
    clip = long_video[start:start + clip_length]

    # 50% warm start (compute hidden from previous frames)
    # 50% cold start (simulate beginning of video)
    if random.random() < 0.5:
        with torch.no_grad():
            warmup_clip = long_video[max(0, start-8):start]
            _, hidden = model(warmup_clip, None)
    else:
        hidden = None

    output, _ = model(clip, hidden)
    return compute_loss(output, ground_truth[start:start + clip_length])
```

### Memory-Efficient Inference Pipeline

```python
@torch.inference_mode()
def process_long_video(model, video_path, output_path, clip_length=16, overlap=4):
    """Stream-based processing - don't load entire video into memory."""
    reader = VideoReader(video_path)
    writer = VideoWriter(output_path, fps=reader.fps)

    hidden = None
    frame_buffer = []

    for frame in reader:
        frame_buffer.append(frame)

        if len(frame_buffer) >= clip_length:
            clip = torch.stack(frame_buffer).to(device)

            with torch.autocast(device_type="mps", dtype=torch.float16):
                output_clip, hidden = model(clip, hidden)

            # Write non-overlapping frames
            for out_frame in output_clip[:-overlap] if overlap else output_clip:
                writer.write(out_frame.cpu())

            frame_buffer = frame_buffer[-overlap:] if overlap else []
            hidden = detach_hidden(hidden)
            torch.mps.empty_cache()

    # Process remaining frames
    if frame_buffer:
        clip = torch.stack(frame_buffer).to(device)
        output_clip, _ = model(clip, hidden)
        for out_frame in output_clip:
            writer.write(out_frame.cpu())

    writer.close()
```

### TL;DR

- Don't process frame-by-frame (too slow, no temporal context)
- Don't load entire video (memory explosion)
- Use **clip-recurrent**: 16-frame clips, 4-frame overlap, hidden state propagation
- Hidden state should contain: previous features, flow, and predicted masks
- Train with warm starts to avoid domain gap

**References:**
- RVRT: Recurrent Video Restoration Transformer
- VideoPainter (SIGGRAPH 2024): ID resampling for 1+ minute consistency

## Experiment Results

### Vimeo Sweep (10k steps, num_frames=2)

Best performers for removing floating text/logos:

| Rank | Exp | Config | Key Factors |
|------|-----|--------|-------------|
| 1 | exp3 | Modulation + LPIPS | Mask guidance + strong perceptual loss |
| 2 | exp6 | Baseline + LPIPS | No mask, just LPIPS |
| 3 | exp8 | Large (64ch) + Modulation + VGG | More capacity compensates for weaker loss |

**Observations:**
- **LPIPS is critical**: Both exp3 and exp6 use LPIPS; exp8 compensates with 4x channels
- **Modulation provides edge**: exp3 slightly better than exp6 — mask guidance helps
- **Larger model promising**: exp8 (64ch + modulation) still training, potentially best
- **Pixel-only insufficient**: exp4 (pixel-only) not in top 3
- **VGG alone not enough**: exp2, exp5, exp7 (VGG perceptual) didn't make top 3

**Interpretation:**
- LPIPS loss naturally focuses on high-frequency anomalies (text edges, logo boundaries)
- Mask-guided modulation provides incremental benefit over LPIPS alone
- Model capacity (exp8: 64 channels) combined with modulation may be optimal

### LaMa and ELIR Experiments

| Exp | Model | Config | Status | Notes |
|-----|-------|--------|--------|-------|
| 18 | LaMa | From scratch, 64ch | Done | FFC blocks provide global receptive field |
| 19 | LaMa | Pretrained + finetune | Done | TorchScript weights load via `torch.jit.load` |
| 20 | ELIR | Pretrained + finetune | Active | Architecture matches original (RRDBNet for MMSE) |
| 21 | ELIR | From scratch, 64ch | Backup | Flow matching loss + mask loss |

**ELIR Training Objectives (exp 21):**
1. **Mask prediction** (standalone): `mask_head` predicts artifact regions from degraded input only
2. **Flow matching**: MSE loss between predicted velocity and target velocity in latent space
   - Sample random t ∈ [0,1], interpolate z_t = (1-t)·z_degraded + t·z_clean
   - Target: v_target = z_clean - z_degraded (straight path)
3. **Composite output**: output = input×(1-mask) + restored×mask

**TensorBoard Metrics:**
- `train/flow` — flow matching loss
- `train/mask` — Dice + L1 mask prediction loss
- `train/inpaint` — auxiliary inpainting loss in mask regions
- `train/total` — combined loss

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

## Architecture Variant: Explicit Composite

An alternative to implicit end-to-end learning is explicit mask → inpaint → composite:

```
┌────────────────────────────────────────────────────────────────┐
│  EXPLICIT COMPOSITE ARCHITECTURE                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  degraded ──┬──► [Encoder] ──► [Decoder] ──► inpainted        │
│             │         │                          │             │
│             │         └──► [Mask Head] ──► mask  │             │
│             │                              │     │             │
│             │                              ▼     ▼             │
│             └────────────────────────► output = degraded × (1-mask)
│                                                + inpainted × mask
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Key difference from current modulation approach:**
- **Current (modulation)**: mask modulates internal features, output is direct prediction
- **Explicit composite**: mask used for final alpha-blending, explicitly preserves clean regions

```python
class ExplicitCompositeNet(nn.Module):
    """
    Predict inpainted image and mask separately, then composite.
    Clean regions are explicitly preserved via alpha blending.
    """
    def __init__(self, base_net):
        super().__init__()
        self.encoder = base_net.encoder
        self.decoder = base_net.decoder
        self.mask_head = nn.Conv2d(channels, 1, 1)  # predict soft mask
        self.inpaint_head = nn.Conv2d(channels, 3, 3, padding=1)  # predict inpainted

    def forward(self, degraded):
        # Encode
        feat, skips = self.encoder(degraded)

        # Decode
        feat = self.decoder(feat, skips)

        # Predict mask and inpainted separately
        mask = torch.sigmoid(self.mask_head(feat))  # [0, 1]
        inpainted = torch.sigmoid(self.inpaint_head(feat))  # full image inpainting

        # Explicit composite: preserve clean regions exactly
        output = degraded * (1 - mask) + inpainted * mask

        return output, mask
```

**Benefits:**
| Aspect | Benefit |
|--------|---------|
| Preservation guarantee | Clean regions (mask=0) pass through unchanged |
| Interpretable | Can visualize what model thinks is artifact vs clean |
| Gradient focus | Mask prediction error directly affects output |
| Easier debugging | Check if mask is correct, check if inpainting is good |

**Training considerations:**
- Mask supervision helps (provides signal for where to inpaint)
- `--mask-weight` focuses pixel loss on artifact regions
- Inpaint head only needs to produce good results where mask > 0

**When to use:**
- Overlay artifacts (logos, text) with clear boundaries
- When interpretability matters
- When you want guaranteed preservation of clean regions

**When NOT to use:**
- Global degradations (noise, compression) where mask is everywhere
- Soft/blended artifacts where hard composite creates edges

## Implementation Roadmap

> **Note**: Keep the existing `TemporalNAFUNet` (v1) intact. New architecture should be a separate class (e.g., `ExplicitCompositeNet`). Add `--net {v1,composite}` flag to `train.py` and `predict.py` to select which network to use.

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
