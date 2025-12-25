#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
IMG="cvlization/hunyuan-video-1-5:latest"

# Use HF cache from host
HF_CACHE="${HOME}/.cache/huggingface"

# Check if models are in cache
if [ ! -d "${HF_CACHE}/hub/models--tencent--HunyuanVideo-1.5" ]; then
    echo "ERROR: Model not found in HuggingFace cache"
    echo ""
    echo "Please download the models first by running:"
    echo "  ./download_models.sh"
    echo ""
    exit 1
fi

# Pass through CUDA_VISIBLE_DEVICES if set in environment
CUDA_ENV=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_ENV="--env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

docker run --rm --gpus=all \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
    --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "HF_HOME=/root/.cache/huggingface" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    ${CUDA_ENV:+"$CUDA_ENV"} \
    --shm-size=16g \
    "$IMG" python3 predict.py --model_path tencent/HunyuanVideo-1.5 "$@"
