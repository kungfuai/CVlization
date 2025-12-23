#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Inputs/Outputs: if CVL set these, great; else Python will default to ./inputs, ./outputs
IMG="${CVL_IMAGE:-surya}"

# Optimized batch sizes for A10 GPU (23GB VRAM)
# Can tune these upward if you have more VRAM
RECOGNITION_BATCH_SIZE="${RECOGNITION_BATCH_SIZE:-32}"
DETECTOR_BATCH_SIZE="${DETECTOR_BATCH_SIZE:-4}"
LAYOUT_BATCH_SIZE="${LAYOUT_BATCH_SIZE:-4}"

# Create datalab cache directory if it doesn't exist (surya stores models here)
mkdir -p "${HOME}/.cache/datalab"

# Mount workspace as writable (predict script writes outputs to /workspace)
docker run --rm --gpus=all \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${HOME}/.cache/datalab,dst=/root/.cache/datalab" \
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  --env "RECOGNITION_BATCH_SIZE=$RECOGNITION_BATCH_SIZE" \
  --env "DETECTOR_BATCH_SIZE=$DETECTOR_BATCH_SIZE" \
  --env "LAYOUT_BATCH_SIZE=$LAYOUT_BATCH_SIZE" \
  "$IMG" python3 predict.py "$@"
