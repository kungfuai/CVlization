#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Image name
IMG="${CVL_IMAGE:-unsloth_llama_3b_sft}"

# Mount workspace as writable (training writes outputs to /workspace)
# Override entrypoint since unsloth image defaults to supervisord
# Run as root to avoid permission issues with mounted volumes
docker run --rm --gpus=all --shm-size 16G \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--entrypoint "" \
	--user root \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
	--mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	-v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
	--env "PYTHONPATH=/cvlization_repo" \
	--env "PYTHONUNBUFFERED=1" \
	--env "HF_HOME=/root/.cache/huggingface" \
	--env "HF_HUB_CACHE=/root/.cache/huggingface/hub" \
	${WANDB_API_KEY:+-e WANDB_API_KEY=$WANDB_API_KEY} \
	"$IMG" \
	bash -c '
		# Clear stale cache and workaround for unsloth psutil bug
		rm -rf /workspace/unsloth_compiled_cache 2>/dev/null
		python train.py "$@"
	' -- "$@"
