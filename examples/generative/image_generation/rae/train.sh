#!/usr/bin/env bash
set -euo pipefail

# Stage 2 training: Diffusion model training with pretrained RAE encoder
# Supports: imagenet (requires DATA_PATH), cifar10, cifar100, animalfaces (auto-download)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Setup centralized caching
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE" "$SCRIPT_DIR/results"

# Image name
IMG="${CVL_IMAGE:-cvlization/rae:latest}"

# Dataset and config selection
DATASET="${DATASET:-cifar10}"

# Select config based on dataset if not explicitly set
if [ -z "${CONFIG:-}" ]; then
    case "$DATASET" in
        cifar10|cifar100|animalfaces)
            CONFIG="configs/training/DiTDH-S_DINOv2-B_cifar.yaml"
            ;;
        *)
            CONFIG="configs/training/DiTDH-S_DINOv2-B.yaml"
            ;;
    esac
fi
DATA_PATH="${DATA_PATH:-}"
RESULTS_DIR="${RESULTS_DIR:-results}"

# Build mount arguments
MOUNT_ARGS=""
DATA_PATH_ARG=""
if [ -n "$DATA_PATH" ]; then
    MOUNT_ARGS="--mount type=bind,src=${DATA_PATH},dst=/data/imagenet,readonly"
    DATA_PATH_ARG="--data-path /data/imagenet"
fi

# Run training with torchrun for DDP
docker run --rm --gpus all --shm-size 32G --ipc=host \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
    ${MOUNT_ARGS:+"$MOUNT_ARGS"} \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "PYTHONUNBUFFERED=1" \
    --env "HF_HOME=/root/.cache/huggingface" \
    ${WANDB_API_KEY:+--env "WANDB_API_KEY=${WANDB_API_KEY}"} \
    ${ENTITY:+--env "ENTITY=${ENTITY}"} \
    ${PROJECT:+--env "PROJECT=${PROJECT}"} \
    "$IMG" \
    torchrun --standalone --nproc_per_node=1 train.py \
        --config "$CONFIG" \
        --dataset "$DATASET" \
        ${DATA_PATH_ARG:+"$DATA_PATH_ARG"} \
        --results-dir "$RESULTS_DIR" \
        --image-size 256 \
        --precision bf16 \
        "$@"
