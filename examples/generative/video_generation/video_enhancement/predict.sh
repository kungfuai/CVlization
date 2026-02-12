#!/bin/bash
# Run video enhancement inference inside Docker container
#
# Usage:
#   ./predict.sh -i watermarked.mp4 -o clean.mp4              # composite (default)
#   ./predict.sh -i watermarked.mp4 -o clean.mp4 --model nafunet
#   ./predict.sh --generate-sample                             # create sample watermarked video
#   CUDA_VISIBLE_DEVICES=1 ./predict.sh -i input.mp4 -o out.mp4

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="cvl-video-enhancement"

# Create cache directory on host if it doesn't exist
CACHE_DIR="${HOME}/.cache/cvlization"
mkdir -p "${CACHE_DIR}/data"
mkdir -p "${CACHE_DIR}/huggingface"

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

# Check for --generate-sample flag
if [ "$1" = "--generate-sample" ]; then
    echo "Generating sample watermarked video..."
    shift
    eval docker run --rm \
        ${GPU_FLAG} \
        --shm-size=16g \
        -v "${CACHE_DIR}:/cvl-cache" \
        -v "${SCRIPT_DIR}:/workspace" \
        -w /workspace \
        "${IMAGE_NAME}:latest" \
        python make_sample.py "$@"
    exit 0
fi

echo "Running video enhancement inference..."
eval docker run --rm \
    ${GPU_FLAG} \
    --shm-size=16g \
    -v "${CACHE_DIR}:/cvl-cache" \
    -v "${SCRIPT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}:latest" \
    python infer.py "$@"
