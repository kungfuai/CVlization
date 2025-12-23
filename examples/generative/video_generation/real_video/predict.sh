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

# Determine GPU count (default: 1 for single GPU mode)
GPU_COUNT="${GPU_COUNT:-1}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

# Info about GPU mode
if [ "$GPU_COUNT" -eq 1 ]; then
    echo "Running in single GPU mode (requires 80GB+ VRAM)"
    echo "For faster inference, use 2+ GPUs: CUDA_VISIBLE_DEVICES=0,1"
elif [ "$GPU_COUNT" -ge 2 ]; then
    echo "Running in multi-GPU mode with $GPU_COUNT GPUs"
fi

# Build CUDA environment passthrough
CUDA_ENV=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_ENV="--env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# Container name from CVL or generate one
CONTAINER_NAME="${CVL_CONTAINER_NAME:-real-video-$(date +%s)}"

# Build docker command arguments
DOCKER_ARGS=(
    --rm
    --name "${CONTAINER_NAME}"
    --gpus=all
    --workdir /workspace
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace/example,readonly"
    --mount "type=bind,src=${MODELS_DIR},dst=/workspace/models"
    --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface"
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly"
    --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace"
)

# Add optional mount for pre-downloaded Wan model
if [ -d "${MODELS_DIR}/Wan2.2-S2V-14B" ]; then
    DOCKER_ARGS+=(--mount "type=bind,src=${MODELS_DIR}/Wan2.2-S2V-14B,dst=/workspace/RealVideo/wan_models/Wan2.2-S2V-14B,readonly")
fi

DOCKER_ARGS+=(
    --env "PYTHONPATH=/cvlization_repo:/workspace/RealVideo"
    --env "HF_HOME=/root/.cache/huggingface"
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512"
    --env "TORCHINDUCTOR_FX_GRAPH_CACHE=1"
    --env "TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_cache"
    --env "NCCL_DEBUG=VERSION"
    --env "CUDA_DEVICE_MAX_CONNECTIONS=1"
)

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    DOCKER_ARGS+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

DOCKER_ARGS+=(
    --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}"
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    --shm-size=32g
    --ipc=host
)

# Run with appropriate command based on GPU count
if [ "$GPU_COUNT" -eq 1 ]; then
    docker run "${DOCKER_ARGS[@]}" "$IMG" \
        python3 /workspace/example/predict.py "$@"
else
    docker run "${DOCKER_ARGS[@]}" "$IMG" \
        torchrun --standalone --nproc_per_node="${GPU_COUNT}" \
            /workspace/example/predict.py "$@"
fi
