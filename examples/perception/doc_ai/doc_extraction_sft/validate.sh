#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-doc_extraction_sft}"
CACHE_DIR="${CVL_HF_CACHE:-$HOME/.cache/huggingface}"
DATA_DIR="${CVL_DATA_DIR:-/data}"

mkdir -p "$SCRIPT_DIR/outputs" "$CACHE_DIR"

if [ "${CVL_NO_DOCKER:-}" = "1" ]; then
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    python3 validate_dataset.py "$@"
    exit 0
fi

docker run --rm \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${CACHE_DIR},dst=/root/.cache/huggingface" \
    $(if [ -d "$DATA_DIR" ]; then printf '%s\n' --mount "type=bind,src=${DATA_DIR},dst=${DATA_DIR}"; fi) \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "PYTHONUNBUFFERED=1" \
    ${DOC_EXTRACTION_SFT_TRAIN_JSONL:+--env DOC_EXTRACTION_SFT_TRAIN_JSONL="$DOC_EXTRACTION_SFT_TRAIN_JSONL"} \
    "$IMG" python3 validate_dataset.py "$@"
