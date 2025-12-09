#!/usr/bin/env bash
# Build the OpenVLA SimplerEnv Docker image
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${CVL_IMAGE:-cvlization/openvla-simplerenv:latest}"

echo "Building image: ${IMAGE_NAME}"
echo "This may take a while (installing SimplerEnv, ManiSkill2, OpenVLA dependencies)..."

docker build \
    --file "${SCRIPT_DIR}/Dockerfile" \
    --tag "${IMAGE_NAME}" \
    "${SCRIPT_DIR}"

echo ""
echo "Build complete: ${IMAGE_NAME}"
echo ""
echo "To run the demo:"
echo "  ./run.sh"
echo ""
echo "Then open http://localhost:8000 in your browser."
