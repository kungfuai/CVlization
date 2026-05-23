#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-cvlization/vlm-omr-per-measure-sft:latest}"
docker build -t "$IMG" "$SCRIPT_DIR" && echo "Build complete: $IMG"
