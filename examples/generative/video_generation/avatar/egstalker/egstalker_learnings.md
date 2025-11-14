# EGSTalker Training and Rendering Learnings

## Overview
EGSTalker is an audio-driven talking head generation system using 3D Gaussian Splatting. This document captures key learnings from training and rendering with MediaPipe face tracking.

## Critical Training Configuration Issues

### IMPORTANT: Configuration File Selection

**⚠️ There are THREE config files in this directory:**

1. **`arguments/args.py`** - Quick-test config (POOR QUALITY - shipped with example)
2. **`arguments/args_default.py`** - Incomplete/incorrect attempt (DO NOT USE - has mistakes)
3. **`arguments/args_original.py`** - ✅ CORRECT original defaults (USE THIS FOR PRODUCTION)

**Use `arguments/args_original.py` for production training!**

### The Quality Problem
The provided `arguments/args.py` config file produces poor quality results because it's a **quick-test configuration**, not meant for production use.

### Parameter Comparison

| Parameter | Quick Test (args.py) | Production (Default) | Impact |
|-----------|---------------------|---------------------|---------|
| `coarse_iterations` | 100 | 3000 | **CRITICAL** - 97% reduction! |
| `densify_from_iter` | 1000 | 500 | High - Starts densification later |
| `densify_until_iter` | 7000 | 15000 | High - 59% shorter densification period |
| `batch_size` | 8 | 1 | Medium - Reduces gradient precision |
| `densify_grad_threshold_coarse` | 0.001 | 0.0002 | Medium - Less aggressive densification |
| `iterations` | 1000 (was 20000) | 30000 | Total training length |

### Why These Parameters Matter

#### 1. Coarse Iterations (100 vs 3000)
- **Purpose**: Establishes the basic 3D structure before refinement
- **Impact**: With only 100 iterations, the model gets only 3% of the needed foundation training
- **Result**: Poor facial structure, unstable geometry

#### 2. Densification Window (1000-7000 vs 500-15000)
- **Purpose**: Gradually grows the number of Gaussian points to capture fine details
- **Impact**:
  - Starts later (1000 vs 500) - misses early structural opportunities
  - Ends earlier (7000 vs 15000) - insufficient point growth
  - Only 6,000 iterations vs 14,500 iterations (59% reduction)
- **Result**: Coarse appearance, missing facial details

#### 3. Batch Size (8 vs 1)
- **Purpose**: Number of samples processed simultaneously
- **Impact**: Higher batch size trades quality for speed
- **Result**: Less precise gradients, reduced convergence quality

## Training Phases

### Phase 1: Coarse Training (Iterations 0-3000)
- Establishes basic 3D structure
- Initial Gaussian point placement
- Rough facial geometry

### Phase 2: Densification (Iterations 500-15000)
- Gaussian points grow from ~478 to several thousand
- Occurs every 100 iterations
- Gradient-based point splitting and cloning
- Captures fine facial details

### Phase 3: Refinement (Iterations 15000-30000)
- No new points added
- Optimizes existing points
- Fine-tunes appearance and deformation

## Training Time Estimates

| Configuration | Iterations | GPU | Duration | Quality |
|--------------|-----------|-----|----------|---------|
| Quick test | 1,000 | A10 (23GB) | ~5-10 min | Poor |
| Extended test | 20,000 | A10 (23GB) | ~2.5 hours | Poor (wrong config) |
| **Production** | **30,000** | **A10 (23GB)** | **~9-10 hours** | **Good** |

## Rendering with Custom Audio

### Audio Requirements
EGSTalker requires **preprocessed audio features**, not raw WAV files:
- Format: `.npy` file (NumPy array)
- Shape: `[num_frames, 16, 29]` for DeepSpeech features
- The render script expects both:
  - `<audio_name>.npy` - preprocessed features
  - `<audio_name>.wav` - raw audio for video muxing

### Creating Short Audio Clips

#### 1. Trim the WAV file:
```bash
ffmpeg -i aud_train.wav -t 5 -acodec copy aud_short.wav
```

#### 2. Trim the audio features:
```python
import numpy as np

# Load full features
aud_features = np.load('data/test_videos/aud_ds.npy')
# aud_features.shape: (7999, 16, 29) for 290.9 seconds

# Calculate frames for 5 seconds
original_duration = 290.9  # seconds
short_duration = 5.0  # seconds
num_frames = int(aud_features.shape[0] * short_duration / original_duration)

# Trim and save
aud_short_features = aud_features[:num_frames]
np.save('data/test_videos/aud_short.npy', aud_short_features)
```

### Rendering Commands

#### Default rendering (uses training audio):
```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace/host -w /workspace/host \
  egstalker:mediapipe \
  python render.py \
    -s data/test_videos \
    --model_path output/test_model \
    --iteration 30000 \
    --batch 8 \
    --configs arguments/args_default.py \
    --skip_train
```

#### Custom audio rendering:
```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace/host -w /workspace/host \
  egstalker:mediapipe \
  python render.py \
    -s data/test_videos \
    --model_path output/test_model \
    --iteration 30000 \
    --batch 8 \
    --configs arguments/args_default.py \
    --custom_aud aud_short.npy \
    --custom_wav aud_short.wav \
    --skip_train \
    --skip_test
```

## Docker Environment

### Image: `egstalker:mediapipe`
- Base: NVIDIA CUDA 11.8 with PyTorch 2.1.2
- Custom modifications: MediaPipe face tracking (Apache 2.0 license)
- GPU: NVIDIA A10 (23GB VRAM)

### Key Points:
- Always use `-w /workspace/host` to work with mounted code
- Original code path `/workspace/egstalker` contains built-in files
- Mount local directory: `-v $(pwd):/workspace/host`

## File Structure

```
egstalker/
├── arguments/
│   ├── args.py              # Quick-test config (POOR QUALITY)
│   └── args_default.py      # Production config (GOOD QUALITY)
├── data/
│   └── test_videos/
│       ├── aud.npy          # Full audio features
│       ├── aud_train.wav    # Training audio (290.9 sec)
│       ├── aud_short.npy    # Short audio features (5 sec)
│       └── aud_short.wav    # Short audio (5 sec)
├── output/
│   └── test_model/
│       ├── point_cloud/
│       │   ├── iteration_1000/
│       │   ├── iteration_20000/
│       │   └── iteration_30000/  # Production checkpoint
│       └── custom/
│           └── ours_30000/
│               └── renders/
└── render.py
```

## Common Pitfalls

### 1. Using Quick-Test Config for Production
- **Problem**: `arguments/args.py` has severely reduced parameters
- **Solution**: Use `arguments/args_default.py` for production training
- **Identification**: Check `coarse_iterations` - if it's 100, it's a test config

### 2. Training from Scratch vs Resuming
- **Problem**: Training always starts from iteration 1, even with existing checkpoints
- **Current behavior**: No built-in resume functionality in the provided code
- **Workaround**: Plan for full training runs

### 3. Rendering Default Behavior
- **Problem**: `render.py` defaults to 1000 frames for train set (render.py:64)
- **Solution**: Use `--custom_aud` for specific durations
- **Note**: Test set uses full dataset length

### 4. Audio Format Confusion
- **Problem**: Render script expects `.npy` files, not `.wav` files for `--custom_aud`
- **Solution**: Always provide preprocessed features (`.npy`) + raw audio (`.wav`)

## Performance Metrics

### Rendering Speed
- **GPU Inference**: 72-93 FPS (pure model inference)
- **Overall**: Near real-time for short clips
- **Note**: "FPS" in logs refers to rendering throughput, not video framerate

### Training Metrics
- **Loss target**: 0.10-0.12 (production quality)
- **PSNR target**: 26-28 dB
- **Point count**: Starts at ~478, grows to several thousand during densification

## Key Lessons

1. **Configuration files matter**: The difference between test and production configs is dramatic
2. **Training is expensive**: Production quality requires ~10 hours on A10 GPU
3. **Coarse training is critical**: 3000 iterations establishes foundation
4. **Densification window is key**: 14,500 iterations of point growth captures details
5. **Batch size affects quality**: Lower batch size (1) gives better results
6. **Audio preprocessing is required**: Can't use raw WAV files directly for custom rendering
7. **Real-time inference**: Once trained, rendering is fast (~72 FPS)

## Recommendations

### For Production Training:
```bash
# Use production config
docker run --rm --gpus all \
  -v $(pwd):/workspace/host -w /workspace/host \
  egstalker:mediapipe \
  python train.py \
    -s data/test_videos \
    --model_path output/test_model_prod \
    --configs arguments/args_default.py
```

### For Quick Testing:
```bash
# Keep original config but understand quality limitations
# Useful for: pipeline testing, debugging, proof-of-concept
docker run --rm --gpus all \
  -v $(pwd):/workspace/host -w /workspace/host \
  egstalker:mediapipe \
  python train.py \
    -s data/test_videos \
    --model_path output/test_quick \
    --configs arguments/args.py
```

## References

- Original repository: `/tmp/EGSTalker/`
- Default parameters: `/tmp/EGSTalker/arguments/__init__.py`
- MediaPipe integration: Apache 2.0 licensed face tracking
- Base model: 3D Gaussian Splatting with temporal deformation networks

---

*Document created: 2025-01-13*
*Model checkpoint: iteration_20000 (poor quality due to test config)*
*Recommended checkpoint: iteration_30000 with args_default.py*
