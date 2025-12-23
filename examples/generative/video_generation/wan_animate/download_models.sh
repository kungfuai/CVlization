#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${WAN_ANIMATE_MODELS_DIR:-${HOME}/.cache/cvlization/models/wan_animate}"
MODEL_NAME="Wan-AI/Wan2.2-Animate-14B"
TARGET_DIR="${MODEL_DIR}/Wan2.2-Animate-14B"

mkdir -p "${MODEL_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    pip install --upgrade "huggingface_hub[cli]"
fi

huggingface-cli download "${MODEL_NAME}" \
    --repo-type model \
    --local-dir "${TARGET_DIR}" \
    --local-dir-use-symlinks False

echo "Downloaded to ${TARGET_DIR}"
