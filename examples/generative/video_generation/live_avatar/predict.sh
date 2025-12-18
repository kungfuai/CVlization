#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(pwd)}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# Run inference
docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace/example" \
    --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
    --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    live_avatar \
    python3 /workspace/predict.py "$@"
