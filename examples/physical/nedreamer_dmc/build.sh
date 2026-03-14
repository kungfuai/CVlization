#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-cvl-nedreamer-dmc}"
TAG="${TAG:-latest}"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
docker build -t "${IMAGE_NAME}:${TAG}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
echo "Build complete: ${IMAGE_NAME}:${TAG}"
