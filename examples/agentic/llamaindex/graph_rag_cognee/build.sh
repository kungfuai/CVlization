#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="${CVL_IMAGE:-llamaindex_graph_rag}"

DOCKER_BUILDKIT=0 docker build \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --tag "${IMAGE_NAME}" \
  "${SCRIPT_DIR}"
