#!/bin/bash
set -e

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get image name from directory
IMAGE_NAME="$(basename "$SCRIPT_DIR")"

# Define cache directory
CACHE_DIR="${HOME}/.cache/cvlization/qwen_image_layered"
mkdir -p "$CACHE_DIR"

PORT="${1:-7870}"

echo "Starting RGBA edit app on port $PORT..."

docker run --rm --gpus=all \
    --user "$(id -u):$(id -g)" \
    -p "$PORT:$PORT" \
    -e GRADIO_SERVER_NAME=0.0.0.0 \
    -e GRADIO_SERVER_PORT="$PORT" \
    -e HF_HOME=/cvl-cache/huggingface \
    -e TRANSFORMERS_CACHE=/cvl-cache/huggingface/transformers \
    -e HF_DATASETS_CACHE=/cvl-cache/huggingface/datasets \
    -v "$CACHE_DIR:/cvl-cache" \
    -v "$SCRIPT_DIR:/workspace" \
    --workdir /workspace \
    "$IMAGE_NAME" \
    python3 tool/edit_rgba_image.py
