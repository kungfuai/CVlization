#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-unsloth/unsloth:latest}"
CACHE_DIR="${CVL_HF_CACHE:-$HOME/.cache/huggingface}"
DATA_DIR="${CVL_DATA_DIR:-/data}"
DOCKER_GPUS="${CVL_DOCKER_GPUS:-all}"

if [ -n "${CVL_GPUS:-}" ]; then
    DOCKER_GPUS="\"device=${CVL_GPUS}\""
fi

mkdir -p "$SCRIPT_DIR/outputs" "$CACHE_DIR"

docker run --rm --gpus "$DOCKER_GPUS" --shm-size 16G \
    --entrypoint /bin/bash \
    --user root \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${CACHE_DIR},dst=/root/.cache/huggingface" \
    $(if [ -d "$DATA_DIR" ]; then printf '%s\n' --mount "type=bind,src=${DATA_DIR},dst=${DATA_DIR}"; fi) \
    --env "PYTHONPATH=/workspace:/cvlization_repo" \
    --env "PYTHONUNBUFFERED=1" \
    ${DOC_EXTRACTION_SFT_TRAIN_JSONL:+--env DOC_EXTRACTION_SFT_TRAIN_JSONL="$DOC_EXTRACTION_SFT_TRAIN_JSONL"} \
    ${PYTORCH_CUDA_ALLOC_CONF:+--env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"} \
    ${PYTORCH_ALLOC_CONF:+--env PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF"} \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    "$IMG" -lc "$*"
