#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Image name
IMG="${CVL_IMAGE:-egstalker}"

# Mount workspace as writable (predict script writes outputs)
# Note: Mount to /workspace/host to avoid conflicts with /workspace/egstalker in the image
docker run --rm --gpus=all \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace/host \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace/host" \
	--mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	--mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
	--env "PYTHONPATH=/workspace/host:/workspace/egstalker:/cvlization_repo" \
	--env "PYTHONUNBUFFERED=1" \
	${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
	"$IMG" \
	python /workspace/host/predict.py "$@"
