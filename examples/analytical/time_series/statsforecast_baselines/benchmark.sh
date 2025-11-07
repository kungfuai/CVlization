#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

IMG="${CVL_IMAGE:-statsforecast_baselines}"
CONTAINER_NAME="${CVL_CONTAINER_NAME:-}"

mkdir -p "${HOME}/.cache/cvlization"

DOCKER_ARGS=("--rm" "--shm-size" "4G" "--workdir" "/workspace"
  "--mount" "type=bind,src=${SCRIPT_DIR},dst=/workspace"
  "--mount" "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly"
  "--mount" "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization")

if [[ -n "$CONTAINER_NAME" ]]; then
  DOCKER_ARGS+=("--name" "$CONTAINER_NAME")
fi

# CPU-based classical models - GPU not needed
if [[ "${CVL_ENABLE_GPU:-0}" == "1" ]]; then
  DOCKER_ARGS+=("--gpus" "all")
fi

DOCKER_ARGS+=("--env" "PYTHONPATH=/cvlization_repo" "$IMG" "python" "benchmark.py" "$@")

# shellcheck disable=SC2068
docker run "${DOCKER_ARGS[@]}"
