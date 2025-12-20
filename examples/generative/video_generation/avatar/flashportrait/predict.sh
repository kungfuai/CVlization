#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-flashportrait}"

mkdir -p "${HOME}/.cache/huggingface"
mkdir -p "${SCRIPT_DIR}/outputs"

docker run --rm --gpus=all \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --shm-size 16G \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace/local" \
    --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
    --env "PYTHONUNBUFFERED=1" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "$IMG" \
    python /workspace/local/predict.py "$@"
