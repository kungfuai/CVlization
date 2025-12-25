#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
IMG="${CVL_IMAGE:-llava-next-video}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
VIDEO_CACHE="${VIDEO_CACHE:-$HOME/.cache/llava_next_video}"
# Find repo root for cvlization package
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

docker run --rm --gpus=all \
  --shm-size 16g \
  --ipc=host \
  --workdir /workspace \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${SCRIPT_DIR}/../test_images,dst=/workspace/test_images,readonly" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${VIDEO_CACHE},dst=/root/.cache/llava_next_video" \
  --env "LLAVA_NEXT_VIDEO_MODEL_ID=${LLAVA_NEXT_VIDEO_MODEL_ID:-llava-hf/LLaVA-NeXT-Video-7B-hf}" \
  --env "LLAVA_NEXT_VIDEO_CACHE=/root/.cache/llava_next_video" \
  --env "PYTHONUNBUFFERED=1" \
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
  --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    --env "PYTHONPATH=/cvlization_repo" \
  "$IMG" python3 predict.py "$@"
