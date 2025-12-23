#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
# Find repo root for cvlization package
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

IMG="${CVL_IMAGE:-lighton-ocr}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

DOCKER_ARGS=(
  --rm
  --gpus=all
  --shm-size 16g
  --ipc=host
  --workdir /workspace
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace"
  --mount "type=bind,src=${SCRIPT_DIR}/../test_images,dst=/workspace/test_images,readonly"
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface"
  --env "LIGHTON_OCR_MODEL_ID=${LIGHTON_OCR_MODEL_ID:-lightonai/LightOnOCR-1B-1025}"
  --env "PYTHONUNBUFFERED=1"
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace"
  --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}"
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    --env "PYTHONPATH=/cvlization_repo" \
)

docker run "${DOCKER_ARGS[@]}" "$IMG" python3 predict.py "$@"
