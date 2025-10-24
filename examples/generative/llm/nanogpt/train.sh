#!/bin/bash
# Works from both repo root and example directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

# In CVL docker mode, workspace is readonly; in standalone mode, it's writable for outputs
WORKSPACE_RO="${CVL_WORK_DIR:+,readonly}"

docker run --runtime nvidia \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace${WORKSPACE_RO}" \
	--mount "type=bind,src=${HOME}/.cache,dst=/root/.cache" \
	--mount "type=bind,src=${REPO_ROOT}/cvlization,dst=/workspace/cvlization,readonly" \
	-e PYTHONUNBUFFERED=1 \
	${CVL_WORK_DIR:+--mount "type=bind,src=${CVL_WORK_DIR},dst=/mnt/cvl/workspace"} \
	${CVL_WORK_DIR:+-e CVL_INPUTS=/mnt/cvl/workspace} \
	${CVL_WORK_DIR:+-e CVL_OUTPUTS=/mnt/cvl/workspace} \
	${WANDB_API_KEY:+-e WANDB_API_KEY=$WANDB_API_KEY} \
	nanogpt \
	python train.py config/train_shakespeare_char.py "$@"
