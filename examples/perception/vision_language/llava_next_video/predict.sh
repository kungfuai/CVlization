#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-llava-next-video}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
VIDEO_CACHE="${VIDEO_CACHE:-$HOME/.cache/llava_next_video}"

docker run --rm --gpus=all \
  --shm-size 16g \
  --ipc=host \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${SCRIPT_DIR}/../test_images,dst=/workspace/test_images,readonly" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${VIDEO_CACHE},dst=/root/.cache/llava_next_video" \
  --env "LLAVA_NEXT_VIDEO_MODEL_ID=${LLAVA_NEXT_VIDEO_MODEL_ID:-llava-hf/LLaVA-NeXT-Video-7B-hf}" \
  --env "LLAVA_NEXT_VIDEO_CACHE=/root/.cache/llava_next_video" \
  --env "PYTHONUNBUFFERED=1" \
  "$IMG" python3 predict.py "$@"
