#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${CVL_IMAGE:-analytical_llatisa}"

docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo "Built image $IMAGE_NAME"
