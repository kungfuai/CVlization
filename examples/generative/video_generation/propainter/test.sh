#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-cvlization/propainter:latest}"

HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
mkdir -p "${HF_CACHE}"

# Lightweight sanity check (no GPU required)
docker run --rm \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace,readonly" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
    --env "PYTHONPATH=/cvlization_repo" \
    "$IMG" \
    python predict.py --help
