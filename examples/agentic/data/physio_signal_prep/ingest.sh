#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
IMAGE_NAME="${CVL_IMAGE:-physio_signal_prep}"

mkdir -p "${HOME}/.cache/cvlization"

if [[ "${CVL_NO_DOCKER:-}" == "1" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
  cd "${SCRIPT_DIR}"
  python scripts/build_sleepedf_cache.py "$@"
else
  docker run --rm \
    ${CVL_RUN_GPU:+--gpus="${CVL_RUN_GPU}"} \
    ${CVL_CONTAINER_NAME:+--name "${CVL_CONTAINER_NAME}"} \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization" \
    --env "PYTHONPATH=/cvlization_repo:/workspace" \
    "${IMAGE_NAME}" \
    python scripts/build_sleepedf_cache.py "$@"
fi
