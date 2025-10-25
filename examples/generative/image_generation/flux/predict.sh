#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Image name
IMG="${CVL_IMAGE:-flux}"

# Create outputs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/outputs"

# Mount workspace and HuggingFace cache
docker run --rm --gpus=all --shm-size 16G \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
	--mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
	--env "CUDA_VISIBLE_DEVICES=0" \
	"$IMG" \
	python generate.py "$@"