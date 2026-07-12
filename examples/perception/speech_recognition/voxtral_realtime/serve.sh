#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${CVL_IMAGE:-cvl-voxtral-realtime}"
MODEL_ID="${MODEL_ID:-mistralai/Voxtral-Mini-4B-Realtime-2602}"
HOST_ADDR="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CONTAINER_NAME="${VOXTRAL_CONTAINER_NAME:-cvl-voxtral-realtime-server}"

# Build image if not present
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image $IMAGE not found, building..."
  bash "${SCRIPT_DIR}/build.sh"
fi

DOCKER_RUN_FLAGS=(--gpus all --ipc=host --shm-size 16g)

if [ "${VOXTRAL_DETACH:-1}" = "1" ]; then
  DOCKER_RUN_FLAGS+=(-d --name "${CONTAINER_NAME}")
  # Remove stale container if it exists
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Removing existing container ${CONTAINER_NAME} ..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  fi
  echo "Starting Voxtral realtime server (${MODEL_ID}) detached on ${HOST_ADDR}:${PORT}"
else
  DOCKER_RUN_FLAGS+=(--rm)
  echo "Starting Voxtral realtime server (${MODEL_ID}) on ${HOST_ADDR}:${PORT}"
fi

docker run "${DOCKER_RUN_FLAGS[@]}" \
  -p "${PORT}:${PORT}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e MODEL_ID="${MODEL_ID}" \
  -e HOST="${HOST_ADDR}" \
  -e PORT="${PORT}" \
  -e VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-45000}" \
  -e VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}" \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  "${IMAGE}"

if [ "${VOXTRAL_DETACH:-1}" = "1" ]; then
  cat <<EOF

Server starting in background. Useful commands:
  Tail logs:   docker logs -f ${CONTAINER_NAME}
  Wait ready:  until curl -fsS http://localhost:${PORT}/v1/models >/dev/null 2>&1; do sleep 2; done
  Stop:        bash ${SCRIPT_DIR}/stop.sh
EOF
fi
