#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

IMAGE_NAME="${CVL_IMAGE:-llamaindex_graph_rag}"

LANCEDB_CACHE="${SCRIPT_DIR}/.cache/lancedb"
mkdir -p "${LANCEDB_CACHE}"

DOCKER_RUN=(
  docker run --rm
  ${CVL_RUN_GPU:+--gpus="${CVL_RUN_GPU}"}
  ${CVL_CONTAINER_NAME:+--name "${CVL_CONTAINER_NAME}"}
  --workdir /workspace
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace"
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly"
  --mount "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization"
  --mount "type=bind,src=${LANCEDB_CACHE},dst=/workspace/.cache/lancedb"
  --env "PYTHONPATH=/cvlization_repo:/workspace"
  --env "HF_HOME=/workspace/.cache/huggingface"
  --env "LANCEDB_DIR=/workspace/.cache/lancedb"
)

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  DOCKER_RUN+=(--env OPENAI_API_KEY="${OPENAI_API_KEY}")
fi

for var in LLAMA_GRAPHRAG_PROVIDER LLAMA_GRAPHRAG_OPENAI_MODEL LLAMA_GRAPHRAG_HF_MODEL LLAMA_GRAPHRAG_EMBED_MODEL; do
  if [[ -n "${!var:-}" ]]; then
    DOCKER_RUN+=(--env "${var}=${!var}")
  fi
done

"${DOCKER_RUN[@]}" "${IMAGE_NAME}" python query.py "$@"
