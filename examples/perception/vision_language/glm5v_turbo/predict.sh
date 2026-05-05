#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Image name
IMG="${CVL_IMAGE:-glm5v-turbo}"

# GLM-5V-Turbo is API-based (no GPU required).
# Pass ZAI_API_KEY to authenticate with Z.ai.
docker run --rm \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${SCRIPT_DIR}/../test_images,dst=/workspace/test_images,readonly" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "ZAI_API_KEY=${ZAI_API_KEY:-}" \
  --env "ZAI_BASE_URL=${ZAI_BASE_URL:-}" \
  --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  "$IMG" python3 predict.py "$@"
