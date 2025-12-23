#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Image name
IMG="${CVL_IMAGE:-wan_animate}"

# Default to GPU 1 unless user overrides
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Model and cache paths
MODELS_DIR="${WAN_ANIMATE_MODELS_DIR:-${HOME}/.cache/cvlization/models/wan_animate}"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"

mkdir -p "${MODELS_DIR}"
mkdir -p "${HF_CACHE}"

# CVL integration: if CVL_WORK_DIR is set, we're being called by 'cvl run'
WORKSPACE_RO="${CVL_WORK_DIR:+,readonly}"

if [[ -z "${CVL_WORK_DIR:-}" ]]; then
    USER_CWD="$(pwd)"
fi

docker run --rm --gpus=all \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace${WORKSPACE_RO}" \
	--mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	--mount "type=bind,src=${MODELS_DIR},dst=/models" \
	--mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
	${CVL_WORK_DIR:+--mount "type=bind,src=${CVL_WORK_DIR},dst=/mnt/cvl/workspace"} \
	${USER_CWD:+--mount "type=bind,src=${USER_CWD},dst=/user_data,readonly"} \
	--env "PYTHONPATH=/cvlization_repo:/opt/Wan2.2" \
	--env "PYTHONUNBUFFERED=1" \
	--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
	${CVL_WORK_DIR:+-e CVL_WORK_DIR=/mnt/cvl/workspace} \
	-e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
  -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
	${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
	"$IMG" \
	python predict.py "$@"
