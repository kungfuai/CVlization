#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Setup centralized caching
mkdir -p "${HOME}/.cache/huggingface" "${HOME}/.cache/torch"

# Image name
IMG="${CVL_IMAGE:-repa}"

# Run generation
docker run --rm --gpus=all --shm-size 8G \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
    --mount "type=bind,src=${HOME}/.cache/torch,dst=/root/.cache/torch" \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "PYTHONUNBUFFERED=1" \
    "$IMG" \
    python generate.py "$@"
