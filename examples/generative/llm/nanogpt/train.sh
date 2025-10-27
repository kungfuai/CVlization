#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Image name
IMG="${CVL_IMAGE:-nanogpt}"

# Check if training data exists, prepare it if not
if [ ! -f "$SCRIPT_DIR/data/shakespeare_char/train.bin" ]; then
	echo "Training data not found. Preparing dataset..."
	echo "Running: python data/shakespeare_char/prepare.py"
	docker run --rm \
		--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
		"$IMG" \
		python data/shakespeare_char/prepare.py
	echo "Dataset preparation complete!"
	echo ""
fi

# Create outputs directory
mkdir -p "$SCRIPT_DIR/logs"

# Mount workspace as writable (training writes logs/checkpoints to /workspace)
docker run --rm --gpus=all --shm-size 16G \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
	--mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	--mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
	--env "PYTHONPATH=/cvlization_repo" \
	--env "PYTHONUNBUFFERED=1" \
	${WANDB_API_KEY:+-e WANDB_API_KEY=$WANDB_API_KEY} \
	"$IMG" \
	python train.py config/train_shakespeare_char.py "$@"
