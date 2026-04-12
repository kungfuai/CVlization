#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-faster-whisper}"

CACHE_ROOT="${HOME}/.cache/cvlization"
HF_CACHE="${HF_HOME:-${CACHE_ROOT}/huggingface}"
TORCH_CACHE="${TORCH_HOME:-${CACHE_ROOT}/torch}"

mkdir -p "${CACHE_ROOT}" "${HF_CACHE}" "${TORCH_CACHE}"

GPU_DEVICE="${GPU_DEVICE:-${CVL_GPU:-}}"
DOCKER_ARGS=(run --rm)
if [ -n "${GPU_DEVICE}" ]; then
  if [ "${GPU_DEVICE}" = "all" ]; then
    DOCKER_ARGS+=(--gpus=all)
  else
    DOCKER_ARGS+=(--gpus "device=${GPU_DEVICE}")
  fi
elif [ "${CVL_USE_GPU:-${USE_GPU:-0}}" = "1" ]; then
  DOCKER_ARGS+=(--gpus=all)
fi
if [ -n "${HF_TOKEN:-}" ]; then
  DOCKER_ARGS+=(--env "HF_TOKEN=${HF_TOKEN}")
fi
if [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
  DOCKER_ARGS+=(--env "HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}")
fi

docker "${DOCKER_ARGS[@]}" \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
  --mount "type=bind,src=${CACHE_ROOT},dst=/root/.cache/cvlization" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${TORCH_CACHE},dst=/root/.cache/torch" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=/root/.cache/huggingface" \
  --env "TORCH_HOME=/root/.cache/torch" \
  --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  "$IMG" python predict.py "$@"
