#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${CLARA_IMAGE:-cvl-clara}"

cd "${SCRIPT_DIR}"
echo "Building ${IMAGE} (torch 2.9.1)..."
docker build -t "${IMAGE}" .
