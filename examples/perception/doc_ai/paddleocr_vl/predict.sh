#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMAGE_NAME="$(basename "$SCRIPT_DIR")"
IMG="${CVL_IMAGE:-$IMAGE_NAME}"

# Create HuggingFace cache directory
mkdir -p "${HOME}/.paddleocr"

docker run --rm --gpus=all \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.paddleocr,dst=/root/.paddleocr" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "CUDA_VISIBLE_DEVICES=0" \
  --shm-size=8g \
  "$IMG" python3 predict.py "$@"
