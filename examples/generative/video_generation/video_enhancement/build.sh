#!/bin/bash
# Build Docker image for video artifact removal

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="cvl-video-enhancement"
TAG="${1:-latest}"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "Context: ${SCRIPT_DIR}"

docker build \
    -t "${IMAGE_NAME}:${TAG}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}"

echo ""
echo "Build complete: ${IMAGE_NAME}:${TAG}"
echo ""
echo "To run:"
echo "  ./run.sh              # Interactive shell"
echo "  ./prepare_data.sh     # Download and prepare Vimeo dataset"
echo "  ./train.sh            # Run training"
