#!/bin/bash
set -e

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get image name from directory
IMAGE_NAME="$(basename "$SCRIPT_DIR")"

# Define cache directory
CACHE_DIR="${HOME}/.cache/cvlization/qwen_image_layered"
mkdir -p "$CACHE_DIR"

IMAGE_PATH="${1:-assets/test_images/1.png}"

echo "Running decomposition..."
echo "Input image: $IMAGE_PATH"

docker run --rm --gpus=all \
    --user "$(id -u):$(id -g)" \
    -e HF_HOME=/cvl-cache/huggingface \
    -e TRANSFORMERS_CACHE=/cvl-cache/huggingface/transformers \
    -e HF_DATASETS_CACHE=/cvl-cache/huggingface/datasets \
    -v "$CACHE_DIR:/cvl-cache" \
    -v "$SCRIPT_DIR:/workspace" \
    --workdir /workspace \
    "$IMAGE_NAME" \
    python3 decompose.py \
        --image "$IMAGE_PATH" \
        "${@:2}"

echo "Decomposition complete!"
echo "Outputs saved to: outputs/"
