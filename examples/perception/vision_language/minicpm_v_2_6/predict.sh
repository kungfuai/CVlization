#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Image name
IMG="${CVL_IMAGE:-minicpm-v-2-6}"

# Mount workspace as writable (predict script writes outputs to /workspace/outputs)
# Also mount test_images directory for shared test images
# Mount HuggingFace cache for model caching across runs
# Load HF_TOKEN from .env file (for gated model access)
ENV_FILE="${REPO_ROOT}/../CVlization/.env"
if [ -f "$ENV_FILE" ]; then
  source "$ENV_FILE"
fi

docker run --rm --gpus=all \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${SCRIPT_DIR}/../test_images,dst=/workspace/test_images,readonly" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  ${HF_TOKEN:+--env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"} \
  "$IMG" python3 predict.py "$@"
