#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
IMG="cvlization/ltx2:latest"

# HuggingFace cache directory
HF_CACHE="${HOME}/.cache/huggingface"
mkdir -p "$HF_CACHE"

# Run inference
docker run --rm --gpus=all \
    --shm-size=16g \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace/local" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
    --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
    ${CUDA_VISIBLE_DEVICES:+--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"} \
    --env "PYTHONPATH=/cvlization_repo:/workspace/local/vendor" \
    --env "CVL_INPUTS=/mnt/cvl/workspace" \
    --env "CVL_OUTPUTS=/mnt/cvl/workspace" \
    --env "HF_HOME=/root/.cache/huggingface" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    "$IMG" python /workspace/local/predict.py "$@"
