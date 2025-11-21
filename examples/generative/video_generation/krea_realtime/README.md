# Krea Realtime Video

Real-time text-to-video generation using the Krea 14B model, distilled from Wan 2.1 for fast inference.

**✅ SDK Integration**: This implementation integrates the official Krea AI SDK (https://github.com/krea-ai/realtime-video) to support the model's custom ModularPipeline format.

## Features

- **Offline Batch Generation** (`predict`): Generate videos from text prompts with ~1 second time-to-first-frame
- **Real-time Streaming** (`serve`): WebSocket server for live video generation with streaming output
- **Fast Inference**: Optimized for 4-6 inference steps achieving real-time performance
- **Video-to-Video**: Transform existing videos using text prompts (SDK feature)
- **LoRA Support**: Compatible with fine-tuned adapters (SDK feature)
- **Streaming Ready**: Architecture supports webcam and canvas inputs (SDK feature)

## Model Details

- **Size**: 14 billion parameters (28.6GB safetensors file)
- **Base**: Distilled from Wan 2.1 14B using Self-Forcing technique
- **Performance**: 11fps @ 4 steps on NVIDIA B200 GPU
- **Optimizations**: KV Cache Recomputation and Attention Bias for error mitigation
- **Format**: Custom modular format, not standard diffusers

## Requirements

- NVIDIA GPU with 40GB+ VRAM (recommended for 14B model)
  - B200, H100, or A100 recommended
  - A10 (23GB VRAM) may work with optimizations
- Docker with NVIDIA Container Toolkit
- 60GB disk space (model + dependencies)
- Official Krea AI SDK (automatically included in Docker image)

## Quick Start

### 1. Build the Docker image

```bash
./build.sh
```

This will build a Docker image with the Krea AI SDK and all dependencies (~25GB).

### 2. Generate video from text (Offline Mode)

```bash
PROMPT="A serene mountain lake at sunset with reflections" ./predict.sh
```

Output will be saved to `outputs/video.mp4`. You can customize generation:

```bash
PROMPT="A person walking through a forest" \
WIDTH=832 \
HEIGHT=480 \
NUM_BLOCKS=9 \
SEED=42 \
./predict.sh
```

### 3. Start real-time streaming server

```bash
./serve.sh
```

Then open http://localhost:8000 in your browser to access the web UI for real-time video generation.

## Usage with CVL CLI

```bash
# Build the container
cvl run krea-realtime build

# Generate video from text (offline batch mode)
PROMPT="A beautiful landscape" cvl run krea-realtime predict

# Start real-time streaming server
cvl run krea-realtime serve
```

## Advanced Options

### Offline Batch Generation (predict)

Run inside the Docker container or customize via environment variables:

```bash
python3 predict.py \
    --prompt "Your prompt here" \
    --output outputs/custom.mp4 \
    --width 832 \
    --height 480 \
    --num-blocks 9 \
    --seed 42 \
    --fps 24 \
    --kv-cache-frames 3
```

**Parameters**:
- `--prompt`: Text description of desired video (required)
- `--output`: Output video path (default: outputs/video.mp4)
- `--width`: Video width in pixels (default: 832)
- `--height`: Video height in pixels (default: 480)
- `--num-blocks`: Number of blocks to generate - each block is ~1 second at 24fps (default: 9)
- `--seed`: Random seed for reproducibility (default: 42)
- `--fps`: Frames per second for output video (default: 24)
- `--kv-cache-frames`: Number of frames for KV cache (default: 3)
- `--config`: Path to SDK config file (default: configs/self_forcing_server_14b.yaml)

### Real-time Streaming Server (serve)

Run inside the Docker container or customize via environment variables:

```bash
python3 serve.py \
    --host 0.0.0.0 \
    --port 8000 \
    --compile \
    --config configs/self_forcing_server_14b.yaml
```

**Parameters**:
- `--host`: Host to bind server to (default: 0.0.0.0)
- `--port`: Port to bind server to (default: 8000)
- `--compile`: Use torch.compile for better performance (slower startup)
- `--config`: Path to SDK config file (default: configs/self_forcing_server_14b.yaml)

**Endpoints**:
- `GET /`: Web UI for interactive video generation
- `GET /health`: Health check endpoint
- `WebSocket /ws`: WebSocket endpoint for streaming generation

## Tips

1. **Optimal Performance**: The model is optimized for 4-6 inference steps per block
2. **Memory Usage**: Requires 40GB+ VRAM for optimal performance. Reduce `--num-blocks` or resolution if OOM occurs
3. **Consistency**: Use the same `--seed` value for reproducible results
4. **Block Duration**: Each block generates approximately 1 second of video at 24fps
5. **Streaming Mode**: Use `serve` preset for real-time interactive generation with web UI
6. **Batch Mode**: Use `predict` preset for offline high-quality video generation

## Model Source

- **HuggingFace**: [krea/krea-realtime-video](https://huggingface.co/krea/krea-realtime-video)
- **Official SDK**: [github.com/krea-ai/realtime-video](https://github.com/krea-ai/realtime-video)
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC-BY-NC-SA-4.0)
- **Blog Post**: [Krea Realtime 14B Technical Details](https://www.krea.ai/blog/krea-realtime-14b)
- **Technique**: Based on Self-Forcing distillation from Wan 2.1 14B

## Troubleshooting

**Out of Memory (OOM)**:
- Reduce `--num-blocks` (each block is ~1 second of video)
- Lower resolution: use `--width 640 --height 384`
- Reduce `--kv-cache-frames` to 2 or 1
- Model requires 40GB+ VRAM for optimal performance

**Model Download Issues**:
- Models are automatically downloaded to `~/.cache/huggingface/`
- Base model (Wan 2.1) and Krea checkpoint total ~30GB
- Ensure sufficient disk space and stable internet connection

**Server Won't Start**:
- Check port 8000 is not in use: `lsof -i :8000`
- Verify GPU is accessible: `nvidia-smi`
- Check Docker container has GPU access: `docker run --rm --gpus=all nvidia/cuda:12.1.0-base nvidia-smi`

**Quality Issues**:
- Increase `--num-blocks` for longer, more coherent videos
- Experiment with different `--seed` values
- Try different prompts with more specific details
- Note: Quality depends on the Self-Forcing distillation, not controllable via guidance scale
