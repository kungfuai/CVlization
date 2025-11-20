# Training EGSTalker with New Video Data

This guide shows how to train EGSTalker on your own video data.

## Prerequisites

- A talking head video (MP4, AVI, or MOV format)
- Video should be 3-10 seconds long for best results
- Face should be clearly visible and centered
- Good lighting and stable camera

## Step 1: Prepare Your Video

Copy your video to the `data/` directory:

```bash
cd /home/ubuntu/zz/CVlization/examples/generative/video_generation/avatar/egstalker
cp /path/to/your/video.mp4 data/your_video.mp4
```

## Step 2: Preprocess the Video with BFM Face Tracking

Run the preprocessing script to extract frames, perform BFM face tracking, and extract audio:

```bash
./preprocess.sh data/your_video.mp4 data/your_subject
```

This will create:
- `data/your_subject/ori_imgs/` - Extracted video frames
- `data/your_subject/torso_imgs/` - Torso region images
- `data/your_subject/parsing/` - Face segmentation masks
- `data/your_subject/track_params.pt` - BFM tracking data (should be ~3GB for good quality)
- `data/your_subject/transforms_train.json` - Camera parameters for training frames
- `data/your_subject/transforms_val.json` - Camera parameters for validation frames
- `data/your_subject/aud.wav` - Extracted audio
- `data/your_subject/aud_ds.npy` - DeepSpeech audio features

**Important**: Verify that `track_params.pt` is large (~3GB). If it's small (< 100MB), BFM tracking failed and you should rerun preprocessing.

## Step 3: Copy Data to Expected Location

EGSTalker expects data in the `datasets/` directory:

```bash
mkdir -p datasets/your_subject
cp -r data/your_subject/* datasets/your_subject/
```

## Step 4: Train the Model

Run training for 10,000 iterations (takes ~30 minutes on A10 GPU):

```bash
./train.sh \
  -s datasets/your_subject \
  --model_path output/your_subject \
  --iterations 10000
```

Monitor the training progress:
- Loss should decrease from ~0.17 to ~0.03
- PSNR should increase from ~20 to ~36
- Number of Gaussians should be ~34,650 (with BFM tracking)

## Step 5: Test Inference

Run inference with custom audio (automatically extracts DeepSpeech features):

```bash
./predict.sh \
  --model_path output/your_subject \
  --reference_path datasets/your_subject \
  --audio_path data/your_audio.wav \
  --output_dir results/your_test
```

**Camera modes:**
- `--camera_mode cycle` - Cycle through training camera poses (default)
- `--camera_mode static` - Fixed camera at first training pose
- `--camera_mode orbit` - Orbit camera around subject

Output video will be saved to `results/your_test/output_custom_10000iter_renders.mov`

## Video Requirements

**Good training videos:**
- 3-10 seconds duration
- 25-30 FPS
- Face occupies 30-50% of frame
- Neutral to slight expressions
- Frontal face angle (±15 degrees)
- Clean audio track

**Avoid:**
- Videos < 2 seconds (insufficient training data)
- Videos > 30 seconds (unnecessary, slower training)
- Extreme expressions or occlusions
- Profile/side views
- Poor lighting or motion blur

## Troubleshooting

### BFM Tracking Failed (Small track_params.pt)

If `track_params.pt` is small (< 100MB), BFM tracking failed:

1. Check face is clearly visible throughout video
2. Try a different video with better lighting
3. Ensure face is frontal (not profile)
4. Check `download_bfm.sh` was run to download BFM models

### Training Produces Poor Results

1. **Check number of Gaussians**: Should be ~34,650. If only ~478, tracking failed and densification never progressed.
2. **Verify preprocessing**: Ensure `track_params.pt` is ~3GB
3. **Check metrics**: Loss should reach < 0.05, PSNR should reach > 35

### Model Architecture Mismatch Error

If you get errors about tensor size mismatch (e.g., 64 vs 512):
- Use `render_direct.sh` instead of `predict.py`
- This loads the saved config from `output/your_subject/cfg_args`

### Out of Memory During Training

Reduce batch size or number of points:
```bash
./train.sh -s datasets/your_subject --model_path output/your_subject --iterations 10000 --batch_size 8
```

## File Structure After Preprocessing

```
data/your_subject/
├── ori_imgs/          # Original video frames
├── torso_imgs/        # Torso region crops
├── parsing/           # Face segmentation masks
├── track_params.pt    # BFM tracking data (~3GB)
├── transforms_train.json
├── transforms_val.json
├── aud.wav            # Extracted audio
└── aud_ds.npy         # DeepSpeech audio features
```

## Training Output Structure

```
output/your_subject/
├── cfg_args          # Saved model configuration
├── point_cloud/
│   └── iteration_10000/
│       └── point_cloud.ply
├── chkpnt10000.pth   # Model checkpoint
└── custom/
    └── ours_10000/
        └── renders/
            └── output_custom_10000iter_renders.mov
```

## Performance Expectations

**Training (A10 GPU):**
- Preprocessing: ~5-10 minutes
- Training 10,000 iterations: ~30 minutes
- Final model size: ~50MB

**Inference:**
- Rendering speed: 50-80 FPS
- Quality: High fidelity with BFM tracking
- Real-time capable on modern GPUs

## Example Commands

Complete workflow example:

```bash
# 1. Preprocess
./preprocess.sh data/my_video.mp4 data/my_subject

# 2. Copy to datasets
mkdir -p datasets/my_subject
cp -r data/my_subject/* datasets/my_subject/

# 3. Train
./train.sh -s datasets/my_subject --model_path output/my_subject --iterations 10000

# 4. Inference with custom audio
./predict.sh \
  --model_path output/my_subject \
  --reference_path datasets/my_subject \
  --audio_path data/custom_speech.wav \
  --output_dir results/test1
```

## Notes

- BFM tracking is essential for quality results (produces ~34,650 Gaussians)
- Training is per-subject: each person needs their own trained model
- First preprocessing run downloads BFM models (~1GB) to `~/.cache/`
- Audio features use DeepSpeech (automatically extracted during inference)
- Custom audio is automatically processed - no manual feature extraction needed
