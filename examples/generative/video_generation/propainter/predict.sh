#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-cvlization/propainter:latest}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
WEIGHTS_DIR="${PROPAINTER_WEIGHTS_DIR:-}"

mkdir -p "${HF_CACHE}"

WORKSPACE_RO="${CVL_WORK_DIR:+,readonly}"
if [[ -z "${CVL_WORK_DIR:-}" ]]; then
    USER_CWD="$(pwd)"
fi

WEIGHTS_MOUNT=""
WEIGHTS_ARG=""
if [[ -n "${WEIGHTS_DIR}" ]]; then
    mkdir -p "${WEIGHTS_DIR}"
    WEIGHTS_MOUNT="--mount type=bind,src=${WEIGHTS_DIR},dst=/weights"
    WEIGHTS_ARG="--weights_dir /weights"
fi

docker run --rm --gpus=all \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace${WORKSPACE_RO}" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    ${WEIGHTS_MOUNT:+"${WEIGHTS_MOUNT}"} \
    --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
    ${CVL_WORK_DIR:+--mount "type=bind,src=${CVL_WORK_DIR},dst=/mnt/cvl/workspace"} \
    ${USER_CWD:+--mount "type=bind,src=${USER_CWD},dst=/user_data,readonly"} \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "PYTHONUNBUFFERED=1" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
    ${CVL_WORK_DIR:+-e CVL_WORK_DIR=/mnt/cvl/workspace} \
    -e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
    -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    "$IMG" \
    python predict.py ${WEIGHTS_ARG:+"${WEIGHTS_ARG}"} "$@"
