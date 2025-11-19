#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMG="${CVL_IMAGE:-lighton-ocr}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

DOCKER_ARGS=(
  --rm
  --gpus=all
  --shm-size 16g
  --ipc=host
  --workdir /workspace
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace"
  --mount "type=bind,src=${SCRIPT_DIR}/../test_images,dst=/workspace/test_images,readonly"
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface"
  --env "LIGHTON_OCR_MODEL_ID=${LIGHTON_OCR_MODEL_ID:-lightonai/LightOnOCR-1B-1025}"
  --env "PYTHONUNBUFFERED=1"
)

docker run "${DOCKER_ARGS[@]}" "$IMG" python3 predict.py "$@"
