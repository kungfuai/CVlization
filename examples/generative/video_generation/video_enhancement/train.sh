#!/bin/bash
# Run training inside Docker container

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

# Pass all arguments to train.py
echo "Starting training..."
docker run --rm \
    --gpus all \
    --shm-size=16g \
    -v "${CACHE_DIR}:/cvl-cache" \
    -v "${SCRIPT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}:latest" \
    python train.py "$@"
