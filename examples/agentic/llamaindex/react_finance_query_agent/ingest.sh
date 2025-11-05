#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

IMAGE_NAME="${CVL_IMAGE:-llamaindex_react_finance}"

docker run --rm ${CVL_RUN_GPU:+--gpus="${CVL_RUN_GPU}"} \
  ${CVL_CONTAINER_NAME:+--name "${CVL_CONTAINER_NAME}"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization" \
  --env "PYTHONPATH=/cvlization_repo:/workspace" \
  --env "HF_HOME=/workspace/.cache/huggingface" \
  ${OPENAI_API_KEY:+--env OPENAI_API_KEY=${OPENAI_API_KEY}} \
  "${IMAGE_NAME}" \
  python ingest.py "$@"
