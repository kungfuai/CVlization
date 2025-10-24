#!/bin/bash
# Works from both repo root and example directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

docker run --runtime nvidia \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	-v "$SCRIPT_DIR:/workspace" \
	-v "$REPO_ROOT/data/container_cache:/root/.cache" \
	-v "$REPO_ROOT/cvlization:/workspace/cvlization" \
	-e PYTHONUNBUFFERED=1 \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	nanogpt \
	python train.py config/train_shakespeare_char.py "$@"
