#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-flashportrait}"

echo "Building Docker image: ${IMG}"
docker build -t "$IMG" -f "${SCRIPT_DIR}/Dockerfile" "$SCRIPT_DIR"

echo ""
echo "Build complete: ${IMG}"
echo "Run ./predict.sh to generate portrait animations"
