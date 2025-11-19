#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-kosmos2-grounded-ocr}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
DATA_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"

docker run --rm --gpus=all \
  --shm-size 16g \
  --ipc=host \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${DATA_CACHE},dst=/root/.cache/huggingface/datasets" \
  --env "HF_DATASETS_CACHE=/root/.cache/huggingface/datasets" \
  --env "PYTHONUNBUFFERED=1" \
  "$IMG" python3 train.py "$@"
