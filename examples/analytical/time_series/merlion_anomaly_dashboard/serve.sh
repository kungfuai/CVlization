#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

IMG="${CVL_IMAGE:-merlion_anomaly_dashboard}"
CONTAINER_NAME="${CVL_CONTAINER_NAME:-}"
HOST_PORT="${MERLION_DASHBOARD_PORT:-8050}"

mkdir -p "${HOME}/.cache/cvlization"

DOCKER_ARGS=("--rm" "--shm-size" "4G" "--workdir" "/workspace"
  "--mount" "type=bind,src=${SCRIPT_DIR},dst=/workspace"
  "--mount" "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly"
  "--mount" "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization"
  "-p" "${HOST_PORT}:8050")

if [[ -n "$CONTAINER_NAME" ]]; then
  DOCKER_ARGS+=("--name" "$CONTAINER_NAME")
fi

CMD_ARGS=("$IMG" "gunicorn" "-b" "0.0.0.0:8050" "merlion.dashboard.server:server")
if [[ $# -gt 0 ]]; then
  CMD_ARGS+=("$@")
fi

DOCKER_ARGS+=("--env" "PYTHONPATH=/cvlization_repo" "${CMD_ARGS[@]}")

# shellcheck disable=SC2068
docker run "${DOCKER_ARGS[@]}"
