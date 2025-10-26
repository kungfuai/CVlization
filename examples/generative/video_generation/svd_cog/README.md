# Stable Video Diffusion (Cog)

Generate videos from images using Stability AI's Stable Video Diffusion model.

## License

[Stable Video License](https://stability.ai/stable-video) - Non-commercial community license.

## Source

Adapted from Replicate's [cog-svd](https://github.com/replicate/cog-svd) repo.

## VRAM Requirements

- **Recommended**: 24GB+ VRAM for optimal performance
- **Minimum**: Works on 23GB GPUs (e.g., A10) with `decoding_t=1` (default)
- **decoding_t parameter**: Controls how many frames are decoded at once
  - `decoding_t=1`: ~20GB peak VRAM (slower, works on 23GB GPUs)
  - `decoding_t=7`: ~22GB peak VRAM (faster, may OOM on 23GB GPUs)
  - `decoding_t=14`: ~24GB peak VRAM (fastest, requires 24GB+ VRAM)

## Usage

### Using CVL

```bash
# Build
cvl run svd-cog build

# Generate video (uses default decoding_t=1)
cvl run svd-cog predict -i input_image=@demo.png

# Fast validation with 2 steps
cvl run svd-cog predict -i input_image=@demo.png -i num_steps=2

# Use higher decoding_t on 24GB+ GPU for faster inference
cvl run svd-cog predict -i input_image=@demo.png -i decoding_t=14
```

### Direct Cog Usage

```bash
cog build
cog predict -i input_image=@demo.png
```

## Parameters

- `input_image`: Input image file
- `num_steps`: Sampling steps (default: 25, lower=faster but lower quality)
- `decoding_t`: Frames decoded at once (default: 1 for VRAM efficiency)
- `video_length`: "14_frames_with_svd" or "25_frames_with_svd_xt"
- `sizing_strategy`: How to resize input ("maintain_aspect_ratio", "crop_to_16_9", "use_image_dimensions")
- `seed`: Random seed (optional)