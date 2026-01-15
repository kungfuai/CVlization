#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="cvlization/ltx2:latest"

echo "Building LTX-2 Docker image..."
docker build -t "$IMG" -f "${SCRIPT_DIR}/Dockerfile" "$SCRIPT_DIR"

echo ""
echo "Build complete: $IMG"
echo "Run inference with: ./predict.sh"
