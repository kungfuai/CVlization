#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Image name
IMG="${CVL_IMAGE:-wan2gp_wan}"

# Mount workspace as writable (predict script writes outputs to /workspace)
docker run --rm --gpus=all \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
	--mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	--mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
	--mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
	--env "PYTHONPATH=/cvlization_repo" \
	--env "PYTHONUNBUFFERED=1" \
	--env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
	--env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
	${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
	"$IMG" \
	python predict.py "$@"
