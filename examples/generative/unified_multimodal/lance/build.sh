#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${CVL_LANCE_IMAGE:-cvl-lance}"

cd "$SCRIPT_DIR"
echo "Building image ${IMAGE} ..."
docker build -t "${IMAGE}" .
