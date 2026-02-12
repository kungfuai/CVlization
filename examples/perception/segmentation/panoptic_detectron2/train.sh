#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

IMG="${CVL_IMAGE:-panoptic_detectron2}"
GPU_ARGS=(--gpus=all)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
	GPU_ARGS=(--gpus "device=${CUDA_VISIBLE_DEVICES}")
fi

mkdir -p "${HOME}/.cache/cvlization"

docker run --rm "${GPU_ARGS[@]}" --shm-size 16G \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
	--mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	--mount "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization" \
	--env "PYTHONPATH=/cvlization_repo" \
	--env "PYTHONUNBUFFERED=1" \
	${WANDB_API_KEY:+-e WANDB_API_KEY=$WANDB_API_KEY} \
	"$IMG" \
	python train.py "$@"
