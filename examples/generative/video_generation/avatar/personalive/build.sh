#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Image name (matches example.yaml)
IMG="${CVL_IMAGE:-personalive}"

# Build from the script's directory
echo "Building Docker image: ${IMG}"
docker build -t "$IMG" -f "${SCRIPT_DIR}/Dockerfile" "$SCRIPT_DIR"

echo ""
echo "Build complete: ${IMG}"
echo "Run ./predict.sh to generate portrait animations"
