#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
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
  export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

docker run --rm \
  "${GPU_ARGS[@]}" \
  --ipc=host \
  --workdir /workspace \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --env "PYTHONUNBUFFERED=1" \
  --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    --env "PYTHONPATH=/cvlization_repo" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  "$IMG" python3 predict.py "$@"
