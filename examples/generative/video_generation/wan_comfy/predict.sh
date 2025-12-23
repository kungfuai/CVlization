#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Image name
IMG="${CVL_IMAGE:-wan_comfy}"

# CVL integration: if CVL_WORK_DIR is set, we're being called by 'cvl run'
# In that case, make workspace readonly and mount user's directory as /mnt/cvl/workspace
WORKSPACE_RO="${CVL_WORK_DIR:+,readonly}"

# Parse arguments to handle paths intuitively (only in standalone mode)
# In CVL mode, paths are handled by the user relative to their current directory
if [[ -z "${CVL_WORK_DIR:-}" ]]; then
    # Standalone mode: convert relative output paths to workspace-relative
    # and capture user's current directory for /user_data mount
    USER_CWD="$(pwd)"
    ARGS=()
    for arg in "$@"; do
        if [[ "${PREV_ARG:-}" == "-o" ]] || [[ "${PREV_ARG:-}" == "--output-dir" ]]; then
            # If it's a relative path, convert it to /workspace/<path>
            if [[ "$arg" != /* ]]; then
                arg="/workspace/$arg"
            fi
        fi
        ARGS+=("$arg")
        PREV_ARG="$arg"
    done
else
    # CVL mode: pass arguments through as-is
    ARGS=("$@")
fi

# Mount workspace as writable (predict script writes outputs to /workspace)
# In CVL mode, also mount user's current directory as /mnt/cvl/workspace
# In standalone mode, mount user's cwd as /user_data for easy file access
docker run --rm --gpus=all \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace${WORKSPACE_RO}" \
	--mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	${CVL_WORK_DIR:+--mount "type=bind,src=${CVL_WORK_DIR},dst=/mnt/cvl/workspace"} \
	${USER_CWD:+--mount "type=bind,src=${USER_CWD},dst=/user_data,readonly"} \
	--mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
	--mount "type=bind,src=${HOME}/.cache/cvlization/models,dst=/root/.cache/models" \
	--env "PYTHONPATH=/cvlization_repo" \
	--env "PYTHONUNBUFFERED=1" \
	${CVL_WORK_DIR:+-e CVL_WORK_DIR=/mnt/cvl/workspace} \
	-e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
    -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
	${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
	"$IMG" \
	python predict.py "${ARGS[@]}"
