#!/bin/bash
# Run Docker container with GPU support and cache mounts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="cvl-video-enhancement"

# Create cache directory on host if it doesn't exist
CACHE_DIR="${HOME}/.cache/cvlization"
mkdir -p "${CACHE_DIR}/data"
mkdir -p "${CACHE_DIR}/torch"
mkdir -p "${CACHE_DIR}/huggingface"

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}:latest" > /dev/null 2>&1; then
    echo "Docker image not found. Building first..."
    "${SCRIPT_DIR}/build.sh"
fi

echo "Starting container..."
echo "  Cache: ${CACHE_DIR} -> /cvl-cache"
echo "  Workspace: ${SCRIPT_DIR} -> /workspace"
echo ""

docker run -it --rm \
    --gpus all \
    --shm-size=16g \
    -v "${CACHE_DIR}:/cvl-cache" \
    -v "${SCRIPT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}:latest" \
    "$@"
