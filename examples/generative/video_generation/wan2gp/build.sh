#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="cvlization/wan2gp:latest"

echo "Building Wan2GP Docker image..."
docker build -t "$IMG" -f "${SCRIPT_DIR}/Dockerfile" "$SCRIPT_DIR"

echo ""
echo "Build complete: $IMG"
echo "Run inference with: ./predict.sh"
