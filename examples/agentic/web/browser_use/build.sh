#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-browser-use}"

echo "Building ${IMG} (python:3.12-slim + browser-use + playwright chromium) ..."
docker build -t "${IMG}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
echo "Done: ${IMG}"
