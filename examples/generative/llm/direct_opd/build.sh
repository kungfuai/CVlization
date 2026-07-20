#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${IMAGE_NAME:-direct_opd}"

echo "Building Direct-OPD training image..."
docker build -t "$IMG" "$SCRIPT_DIR"
echo "Build complete: $IMG"
