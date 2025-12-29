#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
# Find repo root for cvlization package
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

# Image name
IMG="${CVL_IMAGE:-stable-diffusion}"

# Create outputs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/outputs"

# Mount workspace and HuggingFace cache
docker run --rm --gpus=all --shm-size 16G \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
	--mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
	--env "CUDA_VISIBLE_DEVICES=0" \
	--mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
	--env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
	--env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    --env "PYTHONPATH=/cvlization_repo" \
	"$IMG" \
	python generate.py "$@"
