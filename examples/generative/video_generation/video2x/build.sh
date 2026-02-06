#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="cvlization/video2x:latest"

echo "Building video2x Docker image..."
docker build -t "$IMG" -f "${SCRIPT_DIR}/Dockerfile" "$SCRIPT_DIR"

echo ""
echo "Build complete: $IMG"
echo "Run inference with: ./predict.sh"
