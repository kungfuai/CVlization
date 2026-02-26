#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${QED_NANO_IMAGE:-qed-nano}"

cd "$SCRIPT_DIR"
echo "Building ${IMAGE} ..."
docker build -t "${IMAGE}" -f Dockerfile .
echo "Done. Run inference via: bash predict.sh"
