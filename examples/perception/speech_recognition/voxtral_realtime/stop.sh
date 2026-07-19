#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${VOXTRAL_CONTAINER_NAME:-cvl-voxtral-realtime-server}"

if ! docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "No container named ${CONTAINER_NAME} found."
  exit 0
fi

echo "Stopping ${CONTAINER_NAME} ..."
docker stop "${CONTAINER_NAME}" >/dev/null || true
docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
echo "Stopped."
