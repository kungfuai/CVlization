#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-worldcanvas}"

mkdir -p "${SCRIPT_DIR}/outputs"

docker run --rm --gpus=all --shm-size 16G \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
	--mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
	--env "CUDA_VISIBLE_DEVICES=0" \
	"$IMG" \
	python predict.py "$@"
