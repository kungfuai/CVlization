#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
IMG="cvlization/real-video:latest"

# Model and cache paths
MODELS_DIR="${SCRIPT_DIR}/models"
HF_CACHE="${HOME}/.cache/huggingface"

# Work directory (from CVL or default to script dir)
WORK_DIR="${CVL_WORK_DIR:-$(pwd)}"

# Create directories
mkdir -p "${MODELS_DIR}"
mkdir -p "${HF_CACHE}"

# Determine GPU count (default: 2)
GPU_COUNT="${GPU_COUNT:-2}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

# Check minimum GPU requirement
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "ERROR: RealVideo requires at least 2 GPUs (each with 80GB+ VRAM)"
    echo "Current GPU_COUNT: $GPU_COUNT"
    echo ""
    echo "Set CUDA_VISIBLE_DEVICES to specify GPUs, e.g.:"
    echo "  export CUDA_VISIBLE_DEVICES=0,1"
    echo ""
    exit 1
fi

# Build CUDA environment passthrough
CUDA_ENV=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_ENV="--env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# Container name from CVL or generate one
CONTAINER_NAME="${CVL_CONTAINER_NAME:-real-video-$(date +%s)}"

# Build optional mounts for pre-downloaded Wan model
MODEL_MOUNTS=""
if [ -d "${MODELS_DIR}/Wan2.2-S2V-14B" ]; then
    MODEL_MOUNTS="--mount type=bind,src=${MODELS_DIR}/Wan2.2-S2V-14B,dst=/workspace/RealVideo/wan_models/Wan2.2-S2V-14B,readonly"
fi

docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus=all \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR}/examples,dst=/workspace/examples,readonly" \
    --mount "type=bind,src=${MODELS_DIR},dst=/workspace/models" \
    --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
    "${MODEL_MOUNTS}" \
    --env "PYTHONPATH=/cvlization_repo:/workspace/RealVideo" \
    --env "HF_HOME=/root/.cache/huggingface" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512" \
    --env "TORCHINDUCTOR_FX_GRAPH_CACHE=1" \
    --env "TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_cache" \
    --env "NCCL_DEBUG=VERSION" \
    --env "CUDA_DEVICE_MAX_CONNECTIONS=1" \
    "${CUDA_ENV}" \
    --shm-size=32g \
    --ipc=host \
    "$IMG" \
    torchrun --standalone --nproc_per_node="${GPU_COUNT}" \
        predict.py "$@"
