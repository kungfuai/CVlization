#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${VLLM_IMAGE:-cvl-vllm}"
MODEL_ID="${MODEL_ID:-allenai/Olmo-3-7B-Instruct}"
HOST_ADDR="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Stable name lets sibling presets (e.g. agentic/code/opencode_qwen3) find
# the detached server and lets `stop.sh` clean it up.
CONTAINER_NAME="${VLLM_CONTAINER_NAME:-cvl-vllm-server}"

echo "Building ${IMAGE} (torch 2.9.1, vLLM) ..."
docker build -t "${IMAGE}" "${SCRIPT_DIR}"

# Agent mode: opt-in flag that enables the three flags every OpenAI-compatible
# *agent* client (opencode, Aider, OpenHands, ...) needs:
#   - tool_choice: "auto"  needs --enable-auto-tool-choice + --tool-call-parser
#   - Qwen3 reasoning shouldn't bleed into `content`     needs --reasoning-parser
#   - 32 k output cap (opencode hard-codes max_tokens=32000) needs --max-model-len >= 65536
# Tool-call / reasoning parsers default to Qwen3 here. For Llama3-style tool
# JSON, GLM, etc., override VLLM_TOOL_PARSER / VLLM_REASONING_PARSER (set
# either to empty string to drop the flag entirely).
if [ "${VLLM_AGENT_DEFAULTS:-0}" = "1" ]; then
  : "${VLLM_MAX_MODEL_LEN:=65536}"
  : "${VLLM_TOOL_PARSER:=qwen3_xml}"
  : "${VLLM_REASONING_PARSER:=qwen3}"
  AGENT_FLAGS=(--enable-auto-tool-choice)
  [ -n "${VLLM_TOOL_PARSER}" ] && AGENT_FLAGS+=(--tool-call-parser "${VLLM_TOOL_PARSER}")
  [ -n "${VLLM_REASONING_PARSER}" ] && AGENT_FLAGS+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
  # For Qwen3-family models, also default thinking OFF for agentic clients.
  # Agentic loops virtually never want chain-of-thought spam between tool
  # calls -- it doubles latency and produces empty `content` until the
  # thinker terminates. Static-JSON clients (opencode et al.) can't pass
  # chat_template_kwargs per-request, so we set the server default.
  # Override: set VLLM_QWEN3_DISABLE_THINKING=0 to keep thinking on.
  if [ "${VLLM_REASONING_PARSER}" = "qwen3" ] && [ "${VLLM_QWEN3_DISABLE_THINKING:-1}" = "1" ]; then
    # serve.py inside the container parses VLLM_EXTRA_ARGS via shlex.split,
    # which strips bare double-quotes from JSON. Wrap the JSON value in
    # literal single-quotes so shlex preserves the inner ".
    AGENT_FLAGS+=(--default-chat-template-kwargs "'{\"enable_thinking\":false}'")
  fi
  VLLM_EXTRA_ARGS="${AGENT_FLAGS[*]} ${VLLM_EXTRA_ARGS:-}"
  echo "Agent-mode defaults: ${AGENT_FLAGS[*]}   max_model_len=${VLLM_MAX_MODEL_LEN}"
fi

DOCKER_RUN_FLAGS=(--gpus all --ipc=host --shm-size 16g)
if [ "${VLLM_DETACH:-0}" = "1" ]; then
  # Detached: a stable name + restart-on-failure, no --rm so the user can
  # `docker logs ${CONTAINER_NAME}` after exit. Use the sibling `stop`
  # preset (or `docker stop ${CONTAINER_NAME}`) to clean up.
  DOCKER_RUN_FLAGS+=(-d --name "${CONTAINER_NAME}")
  # Replace any stale container with the same name (e.g. left over from a
  # prior crashed run) so we don't trip "name already in use".
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Removing existing container ${CONTAINER_NAME} ..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  fi
  echo "Starting vLLM (${MODEL_ID}) detached on ${HOST_ADDR}:${PORT} as ${CONTAINER_NAME}"
else
  DOCKER_RUN_FLAGS+=(--rm)
  echo "Starting vLLM (${MODEL_ID}) on ${HOST_ADDR}:${PORT}"
fi

docker run "${DOCKER_RUN_FLAGS[@]}" \
  -p "${PORT}:${PORT}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e MODEL_ID="${MODEL_ID}" \
  -e HOST="${HOST_ADDR}" \
  -e PORT="${PORT}" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}" \
  -e TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}" \
  -e VLLM_TP_SIZE="${VLLM_TP_SIZE:-}" \
  -e VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}" \
  -e VLLM_DTYPE="${VLLM_DTYPE:-}" \
  -e VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}" \
  -e VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}" \
  "${IMAGE}"

if [ "${VLLM_DETACH:-0}" = "1" ]; then
  cat <<EOF

Started detached. Useful follow-ups:
  Tail logs:   docker logs -f ${CONTAINER_NAME}
  Wait ready:  until curl -fsS http://localhost:${PORT}/v1/models >/dev/null; do sleep 2; done
  Stop:        bash ${SCRIPT_DIR}/stop.sh   (or: cvl run vllm stop)
EOF
fi
