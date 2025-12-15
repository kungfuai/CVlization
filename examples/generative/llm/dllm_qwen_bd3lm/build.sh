#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-cvlization/dllm-qwen-bd3lm:latest}"

docker build --pull -t "$IMG" "$SCRIPT_DIR"
