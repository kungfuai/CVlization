#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-cvlization/gemma3-4b-sft:latest}"

# Create HuggingFace cache directory
mkdir -p "${HOME}/.cache/huggingface"

# Create outputs directory
mkdir -p "${SCRIPT_DIR}/outputs"

echo "Starting Gemma-3 4B SFT training..."

docker run --rm --gpus=all \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --shm-size=16g \
  "$IMG" python3 train.py "$@"

echo "Training complete! Model saved to: ${SCRIPT_DIR}/outputs/final_model"
