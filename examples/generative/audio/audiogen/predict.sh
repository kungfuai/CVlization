#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-audiogen}"
GPU_DEVICE="${GPU_DEVICE:-${CVL_GPU:-}}"

mkdir -p "${HOME}/.cache/huggingface"
mkdir -p "${HOME}/.cache/torch"
mkdir -p "${WORK_DIR}"

if [ -n "$GPU_DEVICE" ]; then
  GPU_FLAG=(--gpus "device=${GPU_DEVICE}")
else
  GPU_FLAG=(--gpus=all)
fi

ENV_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  ENV_ARGS+=(--env "HF_TOKEN=${HF_TOKEN}")
fi
if [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
  ENV_ARGS+=(--env "HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}")
fi

docker run --rm "${GPU_FLAG[@]}" \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${HOME}/.cache/torch,dst=/root/.cache/torch" \
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  --env "HF_HOME=/root/.cache/huggingface" \
  --env "TORCH_HOME=/root/.cache/torch" \
  "${ENV_ARGS[@]}" \
  "$IMG" python predict.py "$@"
