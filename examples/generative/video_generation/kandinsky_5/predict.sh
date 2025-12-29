#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
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
    --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "HF_HOME=/root/.cache/huggingface" \
    --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    --shm-size=16g \
    "$IMG" python3 predict.py "$@"
