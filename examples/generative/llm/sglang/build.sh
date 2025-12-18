#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${SGLANG_IMAGE:-cvl-sglang}"

cd "$SCRIPT_DIR"
echo "Building ${IMAGE} (torch 2.9.1 + sglang 0.5.6.post2)..."
docker build -t "${IMAGE}" .
