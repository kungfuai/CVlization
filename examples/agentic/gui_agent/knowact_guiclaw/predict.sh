#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

IMG="${CVL_IMAGE:-knowact-guiclaw}"

# VLM endpoint config (OpenAI-compatible)
GUICLAW_BASE_URL="${GUICLAW_BASE_URL:-http://localhost:8000/v1}"
GUICLAW_API_KEY="${GUICLAW_API_KEY:-sk-local}"
GUICLAW_MODEL="${GUICLAW_MODEL:-Qwen/Qwen3.5-35B-A3B}"

# Agent config
GUICLAW_BACKEND="${GUICLAW_BACKEND:-dry-run}"
GUICLAW_MAX_STEPS="${GUICLAW_MAX_STEPS:-15}"

# Docker args
DOCKER_ARGS=(run --rm)
if [ -t 0 ] && [ -t 1 ]; then
    DOCKER_ARGS+=(-it)
fi
DOCKER_ARGS+=(--network host)

docker "${DOCKER_ARGS[@]}" \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace,readonly" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
    --mount "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization" \
    --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
    --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    --env "GUICLAW_BASE_URL=${GUICLAW_BASE_URL}" \
    --env "GUICLAW_API_KEY=${GUICLAW_API_KEY}" \
    --env "GUICLAW_MODEL=${GUICLAW_MODEL}" \
    --env "GUICLAW_BACKEND=${GUICLAW_BACKEND}" \
    --env "GUICLAW_MAX_STEPS=${GUICLAW_MAX_STEPS}" \
    "${IMG}" \
    python predict.py "$@"

echo "artifacts at: ${WORK_DIR}/" >&2
