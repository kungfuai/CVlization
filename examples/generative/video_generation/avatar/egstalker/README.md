# EGSTalker: Real-Time Audio-Driven Talking Head Generation

Audio-driven talking head generation using 3D Gaussian Splatting with MediaPipe-based face tracking and Wav2Vec2 audio features.

## Overview

This example implements EGSTalker, a real-time system for generating photorealistic talking head videos from audio input. This implementation uses MediaPipe for license-free face tracking, replacing the original BFM 2009 model.

**Key Features:**
- MediaPipe Face Mesh tracking (Apache 2.0 license, 468 landmarks)
- Temporal smoothing for stable tracking (exponential moving average α=0.7)
- Wav2Vec2 audio feature extraction (768-dimensional features)
- 3D Gaussian Splatting for real-time rendering
- BiSeNet face parsing for semantic segmentation
- Docker-based workflow with CUDA 11.8

**Key Differences from Original EGSTalker:**
- **Face Tracking**: MediaPipe Face Mesh (open-source, Apache 2.0) replaces BFM 2009 (proprietary)
- **Audio Features**: Simplified Wav2Vec2 extraction replaces DeepSpeech dependencies
- **Stability**: Added temporal smoothing for MediaPipe landmarks (scene/talking_dataset_readers.py:347-360)
- **License**: Fully open-source components

## Quick Start

### Prerequisites

- NVIDIA GPU with 16GB+ VRAM (tested on A10)
- Docker with NVIDIA Container Toolkit
- Input video file with audio

### 1. Build Docker Image

```bash
./build.sh
```

Build time: ~5-10 minutes. The image will be tagged as `egstalker`.

**Build Options:**
```bash
# Fast build (default, uses build cache)
./build.sh

# Verbose build with progress output
VERBOSE=1 ./build.sh

# Control parallel compilation jobs (default: 4)
MAX_JOBS=8 ./build.sh

# Custom CUDA architectures (default: 8.6 for A10 GPU)
# 8.6 = A10, RTX 3090, RTX 3080
# 8.9 = L4, L40, RTX 4090
# 9.0 = H100, H200
docker build --build-arg TORCH_CUDA_ARCH_LIST="8.6;9.0" -t egstalker .
```

### 2. Download Models

**Face Parsing Model (Required):**
```bash
./download_models.sh
```

**BFM Models (Optional - for BFM tracking):**

If you want to use BFM-based face tracking instead of MediaPipe:
```bash
./download_bfm.sh
```

This downloads 7 BFM model files (~353MB) from HuggingFace:
- `01_MorphableModel.mat` (230MB) - Main BFM model
- `BFM09_model_info.mat` (122MB) - BFM 2009 model
- Other supporting files

**Note**: BFM tracking may provide better quality than MediaPipe but requires these additional models.

Download the BiSeNet face parsing model (79999_iter.pth):

```bash
./download_models.sh
```

This downloads the face parsing model (51MB) to `data_utils/face_parsing/79999_iter.pth`.

### 3. Preprocessing

Place your input video in `data/` (e.g., `data/input.mp4`), then run the full preprocessing pipeline:

```bash
# MediaPipe tracking (easier, no BFM download required)
./preprocess.sh data/input.mp4 --tracker mediapipe

# BFM tracking (better quality, requires BFM models from step 2)
./preprocess.sh data/input.mp4 --tracker bfm
```

**Processing time**: ~13-15 minutes for 8000-frame video (5 min @ 30 fps)

**Output**: Creates a dataset directory `data/input/` with extracted audio features, images, landmarks, face parsing masks, background, and torso images.

**Tracker Comparison:**
| Feature | MediaPipe | BFM |
|---------|-----------|-----|
| Setup | No extra downloads | Requires `./download_bfm.sh` |
| Quality | Good (~1,500 Gaussians) | Better (~10,000+ Gaussians) |
| Speed | Fast | Moderate |
| License | Apache 2.0 (open) | Research use |

### 4. Training

```bash
./train.sh -s data/input \
  --model_path output/my_model \
  --configs arguments/args.py
```

**Training time**:
- Default: 40,000 iterations (~8-10 hours on NVIDIA A10)
- Test run: 1,000 iterations (~30-40 minutes on NVIDIA A10)
- Training speed: ~4-5 iterations/second

**Hardware requirements**:
- GPU: NVIDIA A10 (24GB) or equivalent
- VRAM: 16GB minimum
- Training batch size: Configurable in `arguments/args.py`

**Monitor training progress**:
```bash
# Watch training log in real-time
tail -f /tmp/training_log.log

# Check metrics
# Loss should decrease from ~0.24 to ~0.10-0.15
# PSNR should increase from ~20 to ~25-30
```

### 5. Inference

Generate a talking head video from trained model and custom audio:

```bash
./predict.sh \
  --audio_path data/custom_audio.wav \
  --reference_path data/input \
  --model_path output/my_model \
  --output_dir results/
```

**Output**: Rendered video will be saved to `results/` directory with audio synced.

## Preprocessing Pipeline

The preprocessing consists of 9 sequential tasks. You can run individual tasks using `--task N`:

### Task Overview

| Task | Description | Output | Est. Time (8000 frames) |
|------|-------------|--------|-------------------------|
| 1 | Extract audio | `aud.wav` | ~5 sec |
| 2 | Extract audio features (DeepSpeech) | `aud.npy`, `aud_ds.npy` | ~90 sec (1.5 min) |
| 3 | Extract images | `ori_imgs/*.jpg` (8000 files) | ~1 min |
| 4 | Face parsing | `parsing/*.png` (8000 masks) | ~3 min |
| 5 | Extract background | `bc.jpg` | ~5 sec |
| 6 | Extract torso/GT images | `torso_imgs/`, `gt_imgs/` | ~3 min |
| 7 | Extract landmarks | `ori_imgs/*.lms` (8000 files) | ~3 min |
| 8 | Face tracking (MediaPipe) | `track_params.pt` | ~2 min |
| 9 | Save transforms | `transforms_train.json`, `transforms_val.json` | ~5 sec |

**Total preprocessing time**: ~13-15 minutes for 8000-frame video

### Running Individual Tasks

If preprocessing times out or fails on a specific task:

```bash
# Example: Re-run landmark extraction only
./preprocess.sh data/input.mp4 --task 7 --tracker mediapipe
```

### Expected File Structure

After preprocessing `data/input.mp4`, the dataset directory `data/input/` should contain:

```
data/input/
├── video.mp4                    # Original video (symlink or copy)
├── aud.wav                      # Extracted audio
├── aud.npy                      # Audio features [N, 768]
├── aud_ds.npy                   # Audio features (copy)
├── au.csv                       # Action units (facial expressions)
├── bc.jpg                       # Background image
├── track_params.pt              # MediaPipe tracking parameters
├── transforms_train.json        # Training camera/frame metadata
├── transforms_val.json          # Validation camera/frame metadata
├── ori_imgs/                    # 8000 extracted frames
│   ├── 0.jpg, 1.jpg, ...
│   └── 0.lms, 1.lms, ...       # 68 facial landmarks per frame
├── parsing/                     # 8000 face parsing masks
│   └── 0.png, 1.png, ...
├── torso_imgs/                  # 8000 torso images
│   └── 0.png, 1.png, ...
└── gt_imgs/                     # 8000 ground truth images
    └── 0.png, 1.png, ...
```

## Configuration

### Preprocessing Arguments

```bash
python data_utils/process.py <video_path> [OPTIONS]

Arguments:
  video_path              Path to input video file
  --task INT              Specific task to run (1-9), -1 for all tasks (default: -1)
  --tracker STR           Face tracker: 'mediapipe' or 'bfm' (default: 'bfm')
```

### Training Arguments

See `arguments/args.py` for full configuration. Key parameters:

- `--model_path`: Output directory for trained model
- `--iterations`: Total training iterations (default: 40000)
- `--test_iterations`: Iterations for validation
- `--save_iterations`: Iterations for saving checkpoints

## Audio Feature Extraction

This implementation uses **Wav2Vec2** (facebook/wav2vec2-base) for audio feature extraction:

```python
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Extract 768-dimensional features per frame
features = model(audio_input).last_hidden_state  # Shape: [N, 768]
```

**Audio Feature Shape**: `[num_frames, 768]` (2D tensor)

The training code automatically handles both 2D (Wav2Vec2) and 3D (DeepSpeech) audio formats (scene/talking_dataset_readers.py:347-360, 384-394).

## MediaPipe Face Tracking

### Temporal Smoothing

MediaPipe landmarks are stabilized using exponential moving average (EMA):

```python
alpha = 0.7  # Smoothing factor
smoothed_landmarks[i] = alpha * current_landmarks[i] + (1 - alpha) * smoothed_landmarks[i-1]
```

This reduces jitter in face tracking across frames.

### Tracking Output

`track_params.pt` contains:
- `vertices`: Face mesh vertices [N, 478, 3] for all N frames
- Camera parameters and transforms

## Training Details

**Architecture:**
- 3D Gaussian Splatting with deformable neural radiance fields
- Audio-conditioned deformation network
- Voxel-based spatial representation
- Appearance feature extraction from VGG16

**Training Configuration:**
- Optimizer: Adam
- Loss: VGG perceptual loss + L1 + LPIPS
- Training stages: Coarse → Fine
- Checkpoint saving: Best and periodic

**Metrics:**
- PSNR (Peak Signal-to-Noise Ratio)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Rendering FPS

## Performance Optimization

### Training Speed

**Typical performance on NVIDIA A10:**
- Coarse stage: 25-30 it/s (iterations per second)
- Fine stage: 4-5 it/s
- Total training time (30k iterations): 8-10 hours

**Optimization tips:**
1. **Reduce iterations for testing**: Use `--iterations 1000` for quick tests
2. **Adjust batch size**: Default is 1 (coarse) and 8 (fine). Increase fine-stage batch size if you have more VRAM
3. **Use checkpoints**: Training saves checkpoints every 500 iterations - resume from checkpoint if interrupted
4. **Monitor with Weights & Biases**: Add `--use_wandb` flag for detailed metrics tracking

### Preprocessing Speed

**Bottlenecks:**
- Face parsing (Task 4): ~3 minutes for 8000 frames
- Landmark extraction (Task 7): ~3 minutes
- Face tracking (Task 8): ~2 minutes

**Speed improvements:**
1. **Reduce video FPS**: Downsample to 25 FPS before processing
2. **Shorter videos**: Process 5-10 second clips for testing
3. **Run tasks in parallel**: If you have multiple GPUs, process different videos simultaneously

### Inference Speed

**Expected performance:**
- Rendering: 15-20 FPS at 450×450 resolution
- Video generation: 1-2 minutes for 10-second clip
- Memory: ~8GB VRAM for inference

## Resources

**Hardware Requirements:**
- GPU: NVIDIA A10 (24GB) or equivalent
- VRAM: 16GB minimum, 24GB recommended
- Disk: ~10GB for preprocessed data (8000 frames)

**Docker Image:**
- Base: PyTorch 2.1.2 + CUDA 11.8
- Size: ~8GB
- Key dependencies: mediapipe, librosa, transformers, wandb

**Model Sizes:**
- BiSeNet face parsing: 51MB
- Wav2Vec2 base: ~360MB (auto-downloaded)
- VGG16 perceptual loss: ~528MB (auto-downloaded)

## Troubleshooting

### Preprocessing Timeout

If preprocessing times out on large videos, run tasks individually:

```bash
# Run tasks 1-6
docker run --rm --gpus all \
  -v $(pwd):/workspace/host -w /workspace/host \
  egstalker:mediapipe \
  python data_utils/process.py data/test_videos/obama.mp4 --task 1 --tracker mediapipe

# Continue for tasks 2-9
```

### Missing transform JSON files

If `transforms_train.json` and `transforms_val.json` are missing:

```bash
# Run task 9 explicitly
docker run --rm --gpus all \
  -v $(pwd):/workspace/host -w /workspace/host \
  egstalker:mediapipe \
  python data_utils/process.py data/test_videos/obama.mp4 --task 9 --tracker mediapipe
```

### Missing landmark files

If you see errors about missing `.lms` files:

```bash
# Re-run landmark extraction (task 7)
docker run --rm --gpus all \
  -v $(pwd):/workspace/host -w /workspace/host \
  egstalker:mediapipe \
  python data_utils/process.py data/test_videos/obama.mp4 --task 7 --tracker mediapipe
```

Verify completion:
```bash
ls data/test_videos/ori_imgs/*.lms | wc -l  # Should match number of frames
```

### Missing torso images

If training fails with OpenCV errors about missing torso images:

```bash
# Re-run torso extraction (task 6)
docker run --rm --gpus all \
  -v $(pwd):/workspace/host -w /workspace/host \
  egstalker:mediapipe \
  python data_utils/process.py data/test_videos/obama.mp4 --task 6 --tracker mediapipe
```

### Audio shape mismatch

The code has been modified to support both 2D (Wav2Vec2) and 3D (DeepSpeech) audio features. If you encounter shape errors, verify `aud_ds.npy` has shape `[N, 768]`:

```bash
python -c "import numpy as np; print(np.load('data/test_videos/aud_ds.npy').shape)"
```

### CUDA Out of Memory

Reduce batch size or image resolution in training configuration.

### Missing wandb

If you see `ModuleNotFoundError: No module named 'wandb'`, rebuild the Docker image (wandb is included in Dockerfile).

## Implementation Status

### ✅ What Works
- **Docker build**: Fully functional with CUDA 11.8, PyTorch 2.1.2
- **Training pipeline**: Tested and working (coarse + fine stages)
- **Preprocessing**: Script created, ready to use (MediaPipe and BFM modes)
- **Data loading**: Handles both DeepSpeech and Wav2Vec2 audio features
- **Model checkpointing**: Automatic saves during training
- **BFM model download**: Automated script for HuggingFace downloads

### ⚠️ Partially Tested
- **Inference workflow**: Core logic implemented, needs end-to-end testing
- **MediaPipe tracking**: Functional but may produce fewer Gaussians than BFM

### 🔧 Known Limitations

1. **Action Units (AU)**: Currently uses placeholder values (all zeros). For production use, implement proper AU extraction or remove AU dependency from training code.

2. **MediaPipe vs BFM Quality**: MediaPipe provides different landmark topology than BFM 2009. MediaPipe may produce fewer 3D Gaussians (~1,500) compared to BFM-based training (10,000+), resulting in lower quality output. For best results, use BFM tracking with `./download_bfm.sh` and `--tracker bfm`.

3. **Preprocessing Performance**: Face parsing and landmark extraction can be slow on long videos. Consider:
   - Reducing video length/FPS before preprocessing
   - Running tasks in parallel on multiple GPUs
   - Using task-specific execution for recovery from failures

## Current Issues

### Critical: Low Gaussian Count and Poor Quality Output

**Problem**: Training produces significantly fewer 3D Gaussians (~1,500) compared to expected counts (10,000-100,000+), resulting in blurry, low-quality output.

**Root Cause**: The original EGSTalker uses **BFM (Basel Face Model)** for 3D face initialization, while this implementation uses **MediaPipe Face Mesh**. Key differences:

- **MediaPipe Face Mesh**: 478 facial landmarks with shallow 3D structure
  - Z-depth: Only 22.3% of face width (1.42 units depth vs 6.37 units width)
  - Limited volumetric geometry → fewer high-gradient regions during training
  - Spatial density: 6.83 vertices per cubic unit

- **BFM (Original)**: Full 3D morphable face model with proper volumetric structure
  - Provides dense, volumetric facial geometry
  - Better coverage for gradient-based Gaussian densification

**Impact on Training**:
- Gaussian densification occurs during coarse stage (iterations 500-3000) when `split_gs_in_fine_stage=False`
- Densification requires gradient magnitude > 0.0002 (`densify_grad_threshold_coarse`)
- MediaPipe's flat geometry produces insufficient gradients → densification stops prematurely at ~1,500 Gaussians
- Growth pattern: Fast (0-800 iterations) → Slow (800-1,000) → Plateau (1,000-3,000)

**Training Observations**:
- Multiple training runs consistently produce ~1,500-1,600 Gaussians
- Batch size changes have no effect (coarse stage forces batch_size=1 at train.py:83)
- Camera poses verified valid (rotation matrices orthogonal, det=1, visual alignment correct)
- Architecture matches original repo defaults (d_model=512, ffn_hidden=2048)

**Status**: Needs investigation of BFM integration or alternative 3D initialization methods. Current MediaPipe implementation produces functional but low-quality results.

**Related Configuration** (arguments/args.py):
```python
OptimizationParams = dict(
    densify_from_iter = 500,
    densification_interval = 100,
    densify_until_iter = 15000,
    coarse_iterations = 3000,
    densify_grad_threshold_coarse = 0.0002,
    split_gs_in_fine_stage = False,  # Densification only in coarse stage
)
```

**Potential Solutions**:
1. Integrate BFM (Basel Face Model) for initialization (matches original implementation)
2. Increase MediaPipe 3D depth through synthetic depth enhancement
3. Lower `densify_grad_threshold_coarse` (may produce noise/instability)
4. Enable `split_gs_in_fine_stage=True` to continue densification after iteration 3000
5. Investigate alternative 3D face models (FLAME, Dlib 3D, etc.)

## References

- **EGSTalker**: [GitHub Repository](https://github.com/ZhuTianheng/EGSTalker) - 3D Gaussian Splatting for talking heads
- **MediaPipe Face Mesh**: [Google MediaPipe](https://google.github.io/mediapipe/solutions/face_mesh.html) - Apache 2.0 license
- **Wav2Vec2**: [Facebook Research](https://huggingface.co/facebook/wav2vec2-base) - Self-supervised speech representation
- **BiSeNet**: [AD-NeRF Face Parsing](https://github.com/YudongGuo/AD-NeRF) - Face semantic segmentation
- **3D Gaussian Splatting**: [Original Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Real-time radiance field rendering

## Citation

If you use this implementation, please cite the original EGSTalker paper and acknowledge the MediaPipe integration:

```bibtex
@inproceedings{egstalker,
  title={EGSTalker: Real-time Audio-driven Talking Head Generation},
  author={Zhu, Tianheng and others},
  booktitle={Conference},
  year={2024}
}
```

## License

This implementation combines:
- EGSTalker code (original license)
- MediaPipe (Apache 2.0)
- Wav2Vec2 (MIT)
- BiSeNet model weights (research/non-commercial)

Please verify license compatibility for your use case.
