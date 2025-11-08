#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="$(basename "$SCRIPT_DIR")"

echo "Building PaddleOCR-VL Docker image..."
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo "Build complete! Image: $IMAGE_NAME"
