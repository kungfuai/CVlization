#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# Run inference
docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace/example" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
    --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONPATH=/cvlization_repo \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    -e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
  -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    live_avatar \
    python3 /workspace/example/predict.py "$@"
