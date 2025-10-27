#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Image name
IMG="${CVL_IMAGE:-granite_docling_finetune}"

# Default values (can be overridden by environment variables)
TRAIN_DATA="${TRAIN_DATA:-ds4sd/docling-dpbench}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/granite_docling_sft}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LR="${LR:-1e-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
LORA_R="${LORA_R:-16}"

# Create output directory
mkdir -p "$SCRIPT_DIR/$OUTPUT_DIR"

# Mount workspace as writable (training writes outputs to /workspace)
docker run --rm --gpus=all \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "TRAIN_DATA=$TRAIN_DATA" \
  --env "OUTPUT_DIR=$OUTPUT_DIR" \
  --env "BATCH_SIZE=$BATCH_SIZE" \
  --env "GRAD_ACCUM=$GRAD_ACCUM" \
  --env "NUM_EPOCHS=$NUM_EPOCHS" \
  --env "LR=$LR" \
  --env "MAX_SEQ_LEN=$MAX_SEQ_LEN" \
  --env "LORA_R=$LORA_R" \
  --env "MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}" \
  "$IMG" python train.py "$@"
