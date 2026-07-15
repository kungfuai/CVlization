#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-direct_opd}"

# Ensure host cache directories exist
mkdir -p "${HOME}/.cache/huggingface"
mkdir -p "${HOME}/.cache/torch"
mkdir -p "${HOME}/.cache/cvlization"

echo "Starting Direct-OPD training..."
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
	--env "HF_HOME=/root/.cache/huggingface" \
	${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \
	${WANDB_API_KEY:+-e WANDB_API_KEY=$WANDB_API_KEY} \
	"$IMG" \
	python train.py "$@"
