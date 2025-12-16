#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${VLLM_IMAGE:-cvl-vllm}"

cd "$SCRIPT_DIR"
echo "Building image ${IMAGE} ..."
docker build -t "${IMAGE}" .
