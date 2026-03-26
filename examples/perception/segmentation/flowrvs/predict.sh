#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
USER_CWD="$(pwd)"
IMG="${CVL_IMAGE:-cvlization/flowrvs:latest}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$HF_CACHE"

GPU_ARGS=(--gpus=all)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_ARGS+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi

docker run --rm \
  "${GPU_ARGS[@]}" \
  --ipc=host \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
  --mount "type=bind,src=/tmp,dst=/tmp" \
  --mount "type=bind,src=${USER_CWD},dst=/user_data,readonly" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  ${HF_TOKEN:+--env "HF_TOKEN=$HF_TOKEN"} \
  "$IMG" \
  python3 /workspace/predict.py "$@"
