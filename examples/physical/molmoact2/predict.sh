#!/usr/bin/env bash
# Run MolmoAct2 action prediction
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${CVL_IMAGE:-molmoact2-inference:latest}"
CONTAINER_NAME="${CVL_CONTAINER_NAME:-molmoact2-predict}"

docker run --rm -it \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --shm-size 16G \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "${SCRIPT_DIR}/artifacts:/workspace/artifacts" \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    "${IMAGE_NAME}" \
    python inference.py "$@"
