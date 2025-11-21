#!/bin/bash
set -e

# Build Docker image for Krea Realtime video generation via Scope
IMAGE_NAME=${IMAGE_NAME:-"krea_realtime_scope"}
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building Docker image: ${IMAGE_NAME}"
echo "This will take approximately 15-20 minutes..."
echo ""

docker build \
    -t "${IMAGE_NAME}" \
    -f "${EXAMPLE_DIR}/Dockerfile" \
    "${EXAMPLE_DIR}"

echo ""
echo "✅ Build complete: ${IMAGE_NAME}"
echo ""
echo "Next steps:"
echo "  1. Generate video: PROMPT=\"Your prompt here\" ./predict.sh"
echo "  2. Or use CVL: PROMPT=\"Your prompt here\" cvl run krea-realtime-scope predict"
