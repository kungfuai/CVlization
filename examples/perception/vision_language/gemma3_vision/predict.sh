#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Load .env from main repo if it exists
MAIN_REPO="${HOME}/projects/CVlization"
if [ -f "${MAIN_REPO}/.env" ]; then
    export $(grep -v '^#' "${MAIN_REPO}/.env" | xargs)
fi

# Image name
IMG="${CVL_IMAGE:-cvlization/gemma3-vision:latest}"

# HuggingFace cache directory
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# Mount workspace as writable (predict script writes outputs to /workspace/outputs)
docker run --rm --gpus=all \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=/root/.cache/huggingface" \
  ${HF_TOKEN:+--env "HF_TOKEN=${HF_TOKEN}"} \
  "$IMG" python3 predict.py "$@"
