#!/usr/bin/env bash
# Run MolmoAct2 action prediction
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"

# Find repo root for cvlization package (go up 3 levels: molmoact2 > physical > examples > repo root)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

IMAGE_NAME="${CVL_IMAGE:-molmoact2-inference:latest}"

docker run --rm --gpus=all \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --workdir /workspace \
    --shm-size 16G \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
    --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "PYTHONUNBUFFERED=1" \
    --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
    --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    --env "HF_HOME=/root/.cache/huggingface" \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    "${IMAGE_NAME}" \
    python3 predict.py "$@"
