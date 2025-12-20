#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-imtalker}"

# Ensure cache directories exist
mkdir -p "${HOME}/.cache/huggingface"
mkdir -p "${HOME}/.cache/torch"
mkdir -p "${HOME}/.cache/cvlization/imtalker/checkpoints"
mkdir -p "${SCRIPT_DIR}/outputs"

docker run --rm --gpus=all --shm-size 8G \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace/local" \
    --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
    --mount "type=bind,src=${HOME}/.cache/torch,dst=/root/.cache/torch" \
    --mount "type=bind,src=${HOME}/.cache/cvlization/imtalker/checkpoints,dst=/workspace/IMTalker/checkpoints" \
    --env "PYTHONUNBUFFERED=1" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "$IMG" \
    python /workspace/local/predict.py "$@"
