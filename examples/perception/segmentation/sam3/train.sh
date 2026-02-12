#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-sam3}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# Request GPUs by default; respect explicit CUDA_VISIBLE_DEVICES if set
GPU_ARGS=(--gpus=all)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_ARGS+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi

# Load HF_TOKEN from repo .env if present (used for gated checkpoint download)
ENV_FILE="$REPO_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

docker run --rm \
  "${GPU_ARGS[@]}" \
  --shm-size 16g \
  --ipc=host \
  --workdir /workspace \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  ${CVL_DATASET_DIR:+--mount "type=bind,src=${CVL_DATASET_DIR},dst=/data"} \
  --env "PYTHONPATH=/workspace:/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  "$IMG" bash -c "pip install --no-cache-dir -q -r /workspace/requirements-train.txt && python3 train_lora.py \$*" -- "$@"
