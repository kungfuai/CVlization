#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-paligemma2-segmentation}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# Load HF_TOKEN from .env if it exists
ENV_FILE="/home/zsi/projects/CVlization/.env"
if [ -f "$ENV_FILE" ]; then
  export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

docker run --rm --gpus='"device=1"' \
  --shm-size 16g \
  --ipc=host \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${PWD},dst=/mnt/host,readonly" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "CUDA_VISIBLE_DEVICES=0" \
  "$IMG" python3 train.py "$@"
