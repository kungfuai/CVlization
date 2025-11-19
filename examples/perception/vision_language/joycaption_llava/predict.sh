#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMG="${CVL_IMAGE:-joycaption-llava}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

docker run --rm --gpus=all \
  --shm-size 16g \
  --ipc=host \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${SCRIPT_DIR}/../test_images,dst=/workspace/test_images,readonly" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --env "JOYCAPTION_MODEL_ID=${JOYCAPTION_MODEL_ID:-fancyfeast/llama-joycaption-beta-one-hf-llava}" \
  --env "PYTHONUNBUFFERED=1" \
  "$IMG" python3 predict.py "$@"
