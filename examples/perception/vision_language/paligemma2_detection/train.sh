#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-paligemma2-detection}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

docker run --rm --gpus=all \
  --shm-size 16g \
  --ipc=host \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${PWD},dst=/mnt/host,readonly" \
  --env "PYTHONUNBUFFERED=1" \
  "$IMG" python3 train.py "$@"
