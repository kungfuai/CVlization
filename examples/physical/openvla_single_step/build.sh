#!/usr/bin/env bash
# Build the OpenVLA single-step inference Docker image
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-openvla-inference:latest}"

echo "Building OpenVLA inference Docker image..."
echo "Image: ${IMAGE_NAME}"
echo ""

docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"

echo ""
echo "Build complete: ${IMAGE_NAME}"
echo ""
echo "To run inference, use: ./run.sh"
