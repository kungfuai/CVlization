#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

# Image name
IMG="${CVL_IMAGE:-wan_s2v}"

# Model and cache paths
MODELS_DIR="${WAN_S2V_MODELS_DIR:-${HOME}/.cache/cvlization/models/wan_s2v}"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"

mkdir -p "${MODELS_DIR}"
mkdir -p "${HF_CACHE}"
mkdir -p "${SCRIPT_DIR}/outputs"

# CVL integration: if CVL_WORK_DIR is set, we're being called by 'cvl run'
WORKSPACE_RO="${CVL_WORK_DIR:+,readonly}"

if [[ -z "${CVL_WORK_DIR:-}" ]]; then
    USER_CWD="$(pwd)"
fi

docker run --rm --gpus=all --shm-size 16G \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace/local${WORKSPACE_RO}" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${MODELS_DIR},dst=/models" \
    --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
    ${CVL_WORK_DIR:+--mount "type=bind,src=${CVL_WORK_DIR},dst=/mnt/cvl/workspace"} \
    ${USER_CWD:+--mount "type=bind,src=${USER_CWD},dst=/user_data,readonly"} \
    --env "PYTHONPATH=/cvlization_repo:/opt/Wan2.2" \
    --env "PYTHONUNBUFFERED=1" \
    --env "HF_HOME=/root/.cache/huggingface" \
    ${CVL_WORK_DIR:+-e CVL_WORK_DIR=/mnt/cvl/workspace} \
    -e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
    -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    "$IMG" \
    python /workspace/local/predict.py "$@"
