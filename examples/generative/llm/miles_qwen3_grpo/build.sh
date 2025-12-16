#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="cvlization/miles_qwen3_grpo:latest"

echo "Building Miles GRPO training image..."
docker build -t "$IMG" -f "${SCRIPT_DIR}/Dockerfile" "$SCRIPT_DIR"

echo "Build complete: $IMG"
