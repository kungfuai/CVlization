#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
IMG="${CVL_IMAGE:-livetalk}"
GPU_DEVICE="${GPU_DEVICE:-${CVL_GPU:-}}"

CACHE_ROOT="${HOME}/.cache/cvlization"
WEIGHTS_DIR="${CACHE_ROOT}/livetalk/weights"
HF_CACHE="${HF_HOME:-${CACHE_ROOT}/huggingface}"
TORCH_CACHE="${TORCH_HOME:-${CACHE_ROOT}/torch}"

ENV_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  ENV_ARGS+=(--env "HF_TOKEN=${HF_TOKEN}")
fi
if [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
  ENV_ARGS+=(--env "HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}")
fi
if [ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
  ENV_ARGS+=(--env "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}")
fi
ENV_ARGS+=(--env "CVL_EXAMPLE_DIR=/workspace/local")

mkdir -p "${WEIGHTS_DIR}"
mkdir -p "${HF_CACHE}"
mkdir -p "${TORCH_CACHE}"

if [ -n "${GPU_DEVICE}" ]; then
  GPU_FLAG=(--gpus "device=${GPU_DEVICE}")
else
  GPU_FLAG=(--gpus=all)
fi

docker run --rm "${GPU_FLAG[@]}" --shm-size 16G \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --workdir /workspace \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace/local" \
    --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
    --mount "type=bind,src=${CACHE_ROOT},dst=/root/.cache/cvlization" \
    --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
    --mount "type=bind,src=${TORCH_CACHE},dst=/root/.cache/torch" \
    --mount "type=bind,src=${WEIGHTS_DIR},dst=/workspace/LiveTalk/pretrained_checkpoints" \
    --env "PYTHONUNBUFFERED=1" \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "HF_HOME=/root/.cache/cvlization/huggingface" \
    --env "TORCH_HOME=/root/.cache/cvlization/torch" \
    --env "MODEL_BASE=/workspace/LiveTalk/pretrained_checkpoints" \
    --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
    --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    "${ENV_ARGS[@]}" \
    "$IMG" \
    python3 /workspace/predict.py "$@"
