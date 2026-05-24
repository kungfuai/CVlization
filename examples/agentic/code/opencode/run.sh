#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
IMG="${CVL_IMAGE:-opencode}"

# Where opencode should send chat completions. Defaults to the local
# detached vLLM server started by `VLLM_DETACH=1 cvl run vllm serve`.
OPENCODE_BASE_URL="${OPENCODE_BASE_URL:-http://localhost:8000/v1}"
# vLLM accepts any string here unless started with --api-key.
VLLM_API_KEY="${VLLM_API_KEY:-sk-local}"

DOCKER_ARGS=(run --rm)
# Allocate a TTY only when stdin/stdout are TTYs — keeps headless CI
# usage (`opencode run "prompt"`) working from non-tty contexts.
if [ -t 0 ] && [ -t 1 ]; then
  DOCKER_ARGS+=(-it)
fi
# --network host so http://localhost:${vllm_port} resolves to the host
# port that the detached vllm server bound. (Linux only; on macOS, use
# --add-host=host.docker.internal:host-gateway and point baseURL there.)
DOCKER_ARGS+=(--network host)

docker "${DOCKER_ARGS[@]}" \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${WORK_DIR},dst=/workspace" \
  --mount "type=bind,src=${SCRIPT_DIR}/opencode.json,dst=/root/.config/opencode/opencode.json,readonly" \
  --env "OPENCODE_BASE_URL=${OPENCODE_BASE_URL}" \
  --env "VLLM_API_KEY=${VLLM_API_KEY}" \
  "${IMG}" opencode "$@"
