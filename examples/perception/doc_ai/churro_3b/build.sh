#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-cvlization/churro-3b:latest}"

echo "Building Docker image: $IMG"
docker build -t "$IMG" "$SCRIPT_DIR"
echo "Build complete!"
