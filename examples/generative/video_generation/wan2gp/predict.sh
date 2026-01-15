#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# WORK_DIR: user's cwd where outputs are saved
# CVL_WORK_DIR is set by cvl CLI to user's original cwd
WORK_DIR="${CVL_WORK_DIR:-$(pwd)}"

IMG="${CVL_IMAGE:-cvlization/wan2gp:latest}"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
# Model checkpoint cache - persists large model files between runs
CKPT_CACHE="${WAN2GP_CKPT_CACHE:-${HOME}/.cache/wan2gp}"

mkdir -p "${HF_CACHE}" "${CKPT_CACHE}"

docker run --rm --gpus=all \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
	--mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	--mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
	--mount "type=bind,src=${CKPT_CACHE},dst=/root/.cache/wan2gp" \
	--mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
	--env "PYTHONPATH=/cvlization_repo" \
	--env "PYTHONUNBUFFERED=1" \
	--env "PYTHONDONTWRITEBYTECODE=1" \
	--env "HF_HOME=/root/.cache/huggingface" \
	--env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
	--env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
	${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
	"$IMG" \
	python predict.py --checkpoint-dir /root/.cache/wan2gp "$@"
