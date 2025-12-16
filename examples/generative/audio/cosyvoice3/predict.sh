#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-cosyvoice3}"

# Ensure cache directories exist on host
mkdir -p "${HOME}/.cache/huggingface"
mkdir -p "${HOME}/.cache/modelscope"

docker run --rm --gpus=all \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${HOME}/.cache/modelscope,dst=/root/.cache/modelscope" \
  --env "PYTHONPATH=/cvlization_repo:/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=/root/.cache/huggingface" \
  --env "MODELSCOPE_CACHE=/root/.cache/modelscope" \
  "$IMG" python predict.py "$@"
