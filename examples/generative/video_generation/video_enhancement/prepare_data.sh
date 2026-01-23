#!/bin/bash
# Download and prepare Vimeo Septuplet dataset inside Docker
#
# Dataset will be cached at:
#   Host: ~/.cache/cvlization/data/vimeo_septuplet/
#   Docker: /cvl-cache/data/vimeo_septuplet/
#
# The dataset is ~82GB, download may take a while.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="cvl-video-enhancement"

# Create cache directory on host if it doesn't exist
CACHE_DIR="${HOME}/.cache/cvlization"
mkdir -p "${CACHE_DIR}/data"

echo "==================================="
echo "Vimeo Septuplet Dataset Preparation"
echo "==================================="
echo ""
echo "Cache directory (host): ${CACHE_DIR}"
echo "Cache directory (docker): /cvl-cache"
echo ""
echo "This will download ~82GB of data."
echo ""

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}:latest" > /dev/null 2>&1; then
    echo "Docker image not found. Building first..."
    "${SCRIPT_DIR}/build.sh"
fi

# Run data preparation inside Docker
docker run --rm \
    --gpus all \
    -v "${CACHE_DIR}:/cvl-cache" \
    -v "${SCRIPT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}:latest" \
    python vimeo_septuplet.py --prepare --stats

echo ""
echo "==================================="
echo "Dataset preparation complete!"
echo "==================================="
