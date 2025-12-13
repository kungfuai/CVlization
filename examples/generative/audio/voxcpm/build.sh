#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-voxcpm}"

echo "Building $IMG..."
docker build -t "$IMG" -f "${SCRIPT_DIR}/Dockerfile" "$SCRIPT_DIR"
echo "Done: $IMG"
