#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-gemma_scope_2_270m_it}"
HF_CACHE="${CVL_HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
WORKSPACE_RO="${CVL_WORK_DIR:+,readonly}"

# Request GPUs by default; respect explicit CUDA_VISIBLE_DEVICES if set
GPU_ARGS=(--gpus=all)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_ARGS+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi

# Load HF_TOKEN from repo .env if present (existing env takes precedence)
ENV_FILE="$REPO_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

mkdir -p "$SCRIPT_DIR/outputs" "$HF_CACHE"

EXTRA_MOUNTS=()
EXTRA_ENVS=()
if [ -z "${CVL_WORK_DIR:-}" ]; then
  EXTRA_MOUNTS+=(--mount "type=bind,src=${REPO_ROOT},dst=/mnt/cvl/workspace,readonly")
  EXTRA_MOUNTS+=(--mount "type=bind,src=${REPO_ROOT}/cvlization,dst=/workspace/cvlization,readonly")
else
  EXTRA_ENVS+=(--env "PYTHONPATH=/mnt/cvl/workspace:${PYTHONPATH:-}")
  EXTRA_ENVS+=(--env "CVL_OUTPUTS=/mnt/cvl/workspace/examples/generative/llm/gemma_scope_2_270m_it/outputs")
fi

docker run --rm \
  "${GPU_ARGS[@]}" \
  --ipc=host \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace${WORKSPACE_RO}" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=/root/.cache/huggingface" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  "${EXTRA_MOUNTS[@]}" \
  "${EXTRA_ENVS[@]}" \
  ${CVL_WORK_DIR:+--mount "type=bind,src=${CVL_WORK_DIR},dst=/mnt/cvl/workspace"} \
  ${CVL_WORK_DIR:+--env "CVL_INPUTS=/mnt/cvl/workspace"} \
  ${CVL_WORK_DIR:+--env "CVL_OUTPUTS=/mnt/cvl/workspace"} \
  "$IMG" python3 predict.py "$@"
