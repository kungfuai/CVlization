#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Image name
IMG="${CVL_IMAGE:-phi-3-5-vision-instruct}"

# Mount workspace as writable (predict script writes outputs to /workspace/outputs)
# Also mount test_images directory for shared test images
docker run --rm --gpus=all \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${SCRIPT_DIR}/../test_images,dst=/workspace/test_images,readonly" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  "$IMG" python3 predict.py "$@"
