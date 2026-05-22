#!/usr/bin/env bash
# Build the MolmoAct2 inference Docker image
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${CVL_IMAGE:-molmoact2-inference:latest}"

echo "Building MolmoAct2 inference Docker image..."
echo "Image: ${IMAGE_NAME}"
echo ""

docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"

echo ""
echo "Build complete: ${IMAGE_NAME}"
echo ""
echo "To run inference: ./predict.sh"
