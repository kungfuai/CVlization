#!/usr/bin/env bash
# Launch Moshi's WebSocket server for live full-duplex dialogue. Use with
# the upstream JS client (https://github.com/kyutai-labs/moshi/tree/main/client)
# or the bundled terminal client (`python -m moshi.client`).
#
# The server listens on host port ${PORT:-8998} and auto-downloads its
# prebuilt web UI from kyutai/moshi-artifacts on startup, so opening
# http://localhost:${PORT} in a browser is the complete local demo.
set -euo pipefail

IMG="${CVL_IMAGE:-moshi}"
HF_REPO="${MOSHI_HF_REPO:-kyutai/moshiko-pytorch-bf16}"
PORT="${PORT:-8998}"

HF_CACHE="${HF_HOME:-${HOME}/.cache/cvlization/huggingface}"
mkdir -p "${HF_CACHE}"

echo "Starting Moshi server (${HF_REPO}) on 0.0.0.0:${PORT}..."
docker run --rm --gpus=all --ipc=host --shm-size 16g \
  -p "${PORT}:${PORT}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
  "${IMG}" \
  python -m moshi.server --host 0.0.0.0 --port "${PORT}" --hf-repo "${HF_REPO}"
