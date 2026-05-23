#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${LLAMA_CPP_IMAGE:-cvl-llama-cpp}"
TAG="${LLAMA_CPP_TAG:-full-cuda}"

cd "$SCRIPT_DIR"
echo "Building ${IMAGE} (llama.cpp upstream tag: ${TAG}) ..."
docker build -t "${IMAGE}" --build-arg "LLAMA_CPP_TAG=${TAG}" .
