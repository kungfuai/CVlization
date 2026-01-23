# Video Artifact Removal

A deep learning model for removing visual artifacts from videos, optimized for Apple M4 inference.

## Architecture

- **2D U-Net with NAFNet-style blocks** - Efficient and well-supported on Apple Silicon
- **Optional temporal attention** - For video consistency without 3D convolutions
- **~5-10M parameters** - Suitable for real-time inference on M4

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

## License

MIT
