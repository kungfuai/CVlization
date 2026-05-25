#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
IMG="${CVL_IMAGE:-pi}"

# vLLM endpoint (defaults to the local detached server started by
# `VLLM_DETACH=1 VLLM_AGENT_DEFAULTS=1 cvl run vllm serve`).
OPENCODE_BASE_URL="${OPENCODE_BASE_URL:-http://localhost:8000/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-sk-local}"
# Default model — pi resolves with fuzzy match, so 'qwen3' works too.
PI_MODEL="${PI_MODEL:-vllm/Qwen/Qwen3.6-27B}"

DOCKER_ARGS=(run --rm)
if [ -t 0 ] && [ -t 1 ]; then
  DOCKER_ARGS+=(-it)
fi
# --network host so http://localhost:${vllm_port} resolves to the host
# port the detached vllm server bound (Linux only).
DOCKER_ARGS+=(--network host)

# pi reads models.yml from $HOME/.omp/agent/. We bind-mount our bundled
# template there. The provider's apiKey field is `VLLM_API_KEY`, which
# pi will look up as an env-var name first -> uses the env value below.
#
# Patch the baseURL on the fly so a non-default OPENCODE_BASE_URL takes
# effect without editing the bundled models.yml.
GEN_DIR="$(mktemp -d -t cvl-pi-XXXXXX)"
trap 'rm -rf "${GEN_DIR}"' EXIT
sed "s|http://localhost:8000/v1|${OPENCODE_BASE_URL}|g" \
  "${SCRIPT_DIR}/models.yml" > "${GEN_DIR}/models.yml"

docker "${DOCKER_ARGS[@]}" \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /work \
  --mount "type=bind,src=${WORK_DIR},dst=/work" \
  --mount "type=bind,src=${GEN_DIR}/models.yml,dst=/root/.omp/agent/models.yml,readonly" \
  --env "VLLM_API_KEY=${VLLM_API_KEY}" \
  --env "PI_MODEL=${PI_MODEL}" \
  "${IMG}" --model "${PI_MODEL}" "$@"
