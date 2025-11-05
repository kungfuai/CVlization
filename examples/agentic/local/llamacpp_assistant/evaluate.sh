#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

IMAGE_NAME="${CVL_IMAGE:-llamacpp_local_assistant}"

mkdir -p "${HOME}/.cache/cvlization"

docker run --rm \
  ${CVL_RUN_GPU:+--gpus="${CVL_RUN_GPU}"} \
  ${CVL_CONTAINER_NAME:+--name "${CVL_CONTAINER_NAME}"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization" \
  --env "PYTHONPATH=/cvlization_repo:/workspace" \
  --env "LLAMACPP_LLM_PATH=${LLAMACPP_LLM_PATH:-}" \
  --env "LLAMACPP_EMBED_PATH=${LLAMACPP_EMBED_PATH:-}" \
  --env "LLAMACPP_DOWNLOAD_MODELS=${LLAMACPP_DOWNLOAD_MODELS:-1}" \
  "${IMAGE_NAME}" \
  python evaluate.py "$@"
