#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

IMG="${CVL_IMAGE:-lite_avatar}"

GPU_ARGS=()
if [[ "${LITE_AVATAR_USE_GPU:-0}" != "0" ]]; then
    GPU_ARGS=(--gpus=all)
fi

mkdir -p "${SCRIPT_DIR}/outputs"
mkdir -p "${HOME}/.cache/modelscope"
mkdir -p "${HOME}/.cache/huggingface"

docker run --rm "${GPU_ARGS[@]}" \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${HOME}/.cache/modelscope,dst=/root/.cache/modelscope" \
    --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
    --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    "$IMG" \
    python predict.py "$@"
