#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
IMG="${CVL_IMAGE:-cvlization/v3-vsr:latest}"

# Centralized cache
CACHE_ROOT="${HOME}/.cache/cvlization"
V3_CACHE="${CACHE_ROOT}/v3"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$V3_CACHE" "$HF_CACHE"

GPU_DEVICE="${GPU_DEVICE:-${CVL_GPU:-}}"
if [ -n "${GPU_DEVICE}" ]; then
  GPU_ARGS=(--gpus "device=${GPU_DEVICE}")
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_ARGS=(--gpus=all --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
else
  GPU_ARGS=(--gpus=all)
fi

docker run --rm \
  "${GPU_ARGS[@]}" \
  --ipc=host \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
  --mount "type=bind,src=${V3_CACHE},dst=/root/.cache/cvlization/v3" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  "$IMG" \
  python3 /workspace/predict.py "$@"
