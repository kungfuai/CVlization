#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-longcat_video_avatar}"

echo "Building Docker image: $IMG"
docker build -t "$IMG" -f "${SCRIPT_DIR}/Dockerfile" "$SCRIPT_DIR"
echo "Build complete: $IMG"
