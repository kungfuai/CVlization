#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-sam3-finetuning}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# Load HF_TOKEN from .env if it exists
ENV_FILE="$REPO_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

docker run --rm --gpus=all \
  --shm-size 16g \
  --ipc=host \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --env "PYTHONPATH=/cvlization_repo:/opt/sam3" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  ${CUDA_VISIBLE_DEVICES:+--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"} \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  "$IMG" python3 train.py "$@"
