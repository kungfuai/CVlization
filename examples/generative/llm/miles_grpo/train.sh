#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
IMG="cvlization/miles_grpo:latest"

# Ensure cache directories exist
mkdir -p "${HOME}/.cache/huggingface"
mkdir -p "${HOME}/.cache/torch"

# Stop any existing Ray processes to avoid conflicts
echo "Cleaning up any existing Ray processes..."
docker run --rm "$IMG" ray stop --force 2>/dev/null || true

echo "Starting Miles GRPO training..."
docker run --rm \
    --gpus all \
    --shm-size=16g \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
    --mount "type=bind,src=${HOME}/.cache/torch,dst=/root/.cache/torch" \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "PYTHONUNBUFFERED=1" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
    "$IMG" python train.py "$@"

echo "Training complete. Outputs saved to: ${SCRIPT_DIR}/outputs/"
