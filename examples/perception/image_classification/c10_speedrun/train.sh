#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

IMG="${CVL_IMAGE:-cifar10_speedrun}"
PIPELINE="${PIPELINE:-hlb}"
DEVICE="${PIPELINE_DEVICE:-cuda}"

DEFAULT_ARGS=(--pipeline "$PIPELINE")
if [[ -n "${DEVICE:-}" ]]; then
	DEFAULT_ARGS+=(--device "$DEVICE")
fi
if [[ -n "${EPOCHS:-}" ]]; then
	DEFAULT_ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
	DEFAULT_ARGS+=(--batch-size "$BATCH_SIZE")
fi

docker run --rm --gpus=all --shm-size 16G \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
	--mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	--mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
	--mount "type=bind,src=${HOME}/.cache/torch,dst=/root/.cache/torch" \
	--mount "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization" \
	--env "PYTHONPATH=/cvlization_repo" \
	--env "PYTHONUNBUFFERED=1" \
	${WANDB_API_KEY:+-e WANDB_API_KEY=$WANDB_API_KEY} \
	"$IMG" \
	python train.py "${DEFAULT_ARGS[@]}" "$@"
