#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
IMG="${CVL_IMAGE:-browser-use}"

# Defaults that point at the local detached vllm started by
# `VLLM_DETACH=1 VLLM_AGENT_DEFAULTS=1 cvl run vllm serve`.
OPENCODE_BASE_URL="${OPENCODE_BASE_URL:-http://localhost:8000/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-sk-local}"
BROWSER_USE_MODEL="${BROWSER_USE_MODEL:-Qwen/Qwen3.5-9B}"
BROWSER_USE_MAX_STEPS="${BROWSER_USE_MAX_STEPS:-20}"

DOCKER_ARGS=(run --rm)
if [ -t 0 ] && [ -t 1 ]; then
  DOCKER_ARGS+=(-it)
fi
# --network host so http://localhost:8000 inside the container reaches
# the host vllm port (Linux-only).
DOCKER_ARGS+=(--network host)
# Browser containers like ample /dev/shm so Chromium doesn't crash.
DOCKER_ARGS+=(--shm-size=1g)

docker "${DOCKER_ARGS[@]}" \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
  --env "OPENCODE_BASE_URL=${OPENCODE_BASE_URL}" \
  --env "VLLM_API_KEY=${VLLM_API_KEY}" \
  --env "BROWSER_USE_MODEL=${BROWSER_USE_MODEL}" \
  --env "BROWSER_USE_MAX_STEPS=${BROWSER_USE_MAX_STEPS}" \
  "${IMG}" "$@"
