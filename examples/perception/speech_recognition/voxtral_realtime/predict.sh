#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-cvl-voxtral-realtime}"
CONTAINER_NAME="${VOXTRAL_CONTAINER_NAME:-cvl-voxtral-realtime-server}"
PORT="${PORT:-8000}"
HOST="${VOXTRAL_HOST:-localhost}"

CACHE_ROOT="${HOME}/.cache/cvlization"
HF_CACHE="${HF_HOME:-${CACHE_ROOT}/huggingface}"

mkdir -p "${CACHE_ROOT}" "${HF_CACHE}"

# Check that the server container is running
if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "ERROR: Server container '${CONTAINER_NAME}' is not running."
  echo "Start it with: bash ${SCRIPT_DIR}/serve.sh"
  echo "Or use test.sh for a full serve→predict→stop cycle."
  exit 1
fi

# Wait for the server to be ready (up to 300s for model loading)
echo "Waiting for server to be ready at http://${HOST}:${PORT}/v1/models ..."
TIMEOUT=300
ELAPSED=0
until curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do
  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "ERROR: Server did not become ready within ${TIMEOUT}s"
    echo "Check logs: docker logs ${CONTAINER_NAME}"
    exit 1
  fi
  sleep 3
  ELAPSED=$((ELAPSED + 3))
done
echo "Server ready (waited ${ELAPSED}s)."

# Run the WebSocket streaming client from within Docker
# The client connects to the host's port via --network=host
docker run --rm \
  --network=host \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
  --mount "type=bind,src=${CACHE_ROOT},dst=/root/.cache/cvlization" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=/root/.cache/huggingface" \
  --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  --env "VOXTRAL_HOST=${HOST}" \
  --env "VOXTRAL_PORT=${PORT}" \
  ${HF_TOKEN:+--env "HF_TOKEN=${HF_TOKEN}"} \
  ${HUGGINGFACE_TOKEN:+--env "HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}"} \
  "$IMG" python predict.py --host "${HOST}" --port "${PORT}" "$@"
