#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Docker image name
IMG="${CVL_IMAGE:-cvlization/gemma3-4b-sft:latest}"

echo "Building Docker image: $IMG"
docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t "$IMG" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "$SCRIPT_DIR"

echo "Build complete: $IMG"
