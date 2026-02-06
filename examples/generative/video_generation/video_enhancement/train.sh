#!/bin/bash
# Run training inside Docker container
#
# Usage:
#   ./train.sh --vimeo --epochs 10          # Use all GPUs
#   CUDA_VISIBLE_DEVICES=1 ./train.sh ...   # Use GPU 1 only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="cvl-video-enhancement"

# Create cache directory on host if it doesn't exist
CACHE_DIR="${HOME}/.cache/cvlization"
mkdir -p "${CACHE_DIR}/data"

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}:latest" > /dev/null 2>&1; then
    echo "Docker image not found. Building first..."
    "${SCRIPT_DIR}/build.sh"
fi

# Handle GPU selection
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    GPU_FLAG="--gpus '\"device=${CUDA_VISIBLE_DEVICES}\"'"
    echo "Using GPU(s): ${CUDA_VISIBLE_DEVICES}"
else
    GPU_FLAG="--gpus all"
    echo "Using all GPUs"
fi

# Pass all arguments to train.py
echo "Starting training..."
echo "TensorBoard: run ./tensorboard.sh in another terminal"
eval docker run --rm \
    ${GPU_FLAG} \
    --shm-size=16g \
    -v "${CACHE_DIR}:/cvl-cache" \
    -v "${SCRIPT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}:latest" \
    python train.py "$@"
