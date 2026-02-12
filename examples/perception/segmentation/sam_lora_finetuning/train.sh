#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-sam-lora-finetuning}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
CVL_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/cvlization"
mkdir -p "$CVL_CACHE"

# Load env vars from .env if it exists
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

docker run --rm --gpus=all \
  --shm-size 16g \
  --ipc=host \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${CVL_CACHE},dst=/root/.cache/cvlization" \
  --env "PYTHONPATH=/workspace" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
  ${CUDA_VISIBLE_DEVICES:+--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"} \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  "$IMG" python3 -m src.training.training_session "$@"
