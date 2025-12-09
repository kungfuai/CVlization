#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
IMG="cvlization/kandinsky-5:latest"

# Use centralized caches from host
HF_CACHE="${HOME}/.cache/huggingface"
WEIGHTS_CACHE="${HOME}/.cache/cvlization/kandinsky5"
mkdir -p "${HF_CACHE}"
mkdir -p "${WEIGHTS_CACHE}"
mkdir -p "${SCRIPT_DIR}/outputs"

docker run --rm --gpus=all \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR}/outputs,dst=/workspace/outputs" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
    --mount "type=bind,src=${WEIGHTS_CACHE},dst=/workspace/weights" \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "HF_HOME=/root/.cache/huggingface" \
    --shm-size=16g \
    "$IMG" python3 predict.py "$@"
