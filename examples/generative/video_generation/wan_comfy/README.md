# WAN 2.1 Video Generation

Generate videos from images using the WAN 2.1 model with text-guided diffusion.

⚠️ **Note:** This example requires a GPU with at least 24GB VRAM. There may be OOM issues when generating videos on smaller GPUs.

## Quick Start

### Setup

Models and tokenizers are downloaded automatically on first run. Alternatively, download them manually:

```bash
# Download model weights (~23 GB) - saved to ~/.cache/cvlization/models/wan/
bash examples/generative/video_generation/wan_comfy/download_models.sh

# Tokenizers auto-download from HuggingFace on first use
# They cache to ~/.cache/huggingface/ automatically
```

### Build the Docker Image

```bash
cvl run generative/video_generation/wan_comfy build
# or
bash examples/generative/video_generation/wan_comfy/build.sh
```

### Generate a Video

**Using CVL CLI (Recommended):**
```bash
cd ~/my_project
cvl run generative/video_generation/wan_comfy predict \
  -p "a beautiful sunset over the ocean" \
  -i my_photo.jpg \
  -o output \
  --steps 10
```

**Direct bash execution:**
```bash
bash examples/generative/video_generation/wan_comfy/predict.sh \
  -p "a beautiful sunset over the ocean" \
  -i /user_data/my_photo.jpg \
  -o my_videos \
  --steps 10
```

## Usage

### Two Ways to Run

#### Option 1: CVL CLI (Recommended)

Paths are relative to your current directory:

```bash
cd ~/my_project
cvl run generative/video_generation/wan_comfy predict \
  -p "a dragon flying through clouds" \
  -i sunset.jpg \
  -o generated_videos \
  --steps 20
```

**Path behavior:**
- Input/output paths are relative to your current directory
- Your current directory is mounted as `/mnt/cvl/workspace`
- Outputs save to your current directory (e.g., `./generated_videos/`)

#### Option 2: Direct Bash

More explicit path handling:

```bash
cd ~/my_project
bash examples/generative/video_generation/wan_comfy/predict.sh \
  -p "a dragon flying through clouds" \
  -i /user_data/sunset.jpg \
  -o my_videos \
  --steps 20
```

**Path behavior:**
- Your current directory is mounted as `/user_data`
- Relative output paths save to the wan_comfy directory
- Use `/user_data/` prefix to access files from your current directory

### Parameters

```
-p, --prompt          Positive prompt for video generation
-n, --negative-prompt Negative prompt (what to avoid)
-i, --reference-image Path to reference image
-o, --output-dir      Output directory (relative or absolute)
--steps              Number of sampling steps (default: 20)
--cfg                Classifier-free guidance scale (default: 6.0)
--fps                Frames per second (default: 16)
--width              Video width (default: 512)
--height             Video height (default: 512)
--length             Number of frames (default: 33)
--seed               Random seed for reproducibility
```

### Example Workflows

**Quick test (5 steps):**
```bash
cvl run generative/video_generation/wan_comfy predict \
  -p "a cat playing piano" \
  -i my_image.jpg \
  -o test_output \
  --steps 5
```

**High quality (50 steps):**
```bash
cvl run generative/video_generation/wan_comfy predict \
  -p "epic mountain landscape with flowing waterfalls" \
  -n "blurry, low quality, static" \
  -i landscape.jpg \
  -o high_quality \
  --steps 50 \
  --cfg 7.5
```

## Performance

- **First run:** Downloads ~23GB of models (one-time)
- **Generation time:** ~2-5 minutes depending on steps and hardware
- **Recommended:** Reduce `--steps` for faster generation (minimum: 5)
- **GPU requirement:** 24GB+ VRAM recommended

## Reference

- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [ComfyUI-Wan](https://comfyanonymous.github.io/ComfyUI_examples/wan/)

