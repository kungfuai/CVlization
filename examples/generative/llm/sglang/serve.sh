#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${SGLANG_IMAGE:-cvl-sglang}"
MODEL_ID="${MODEL_ID:-allenai/Olmo-3-7B-Instruct}"
HOST_ADDR="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"

echo "Building ${IMAGE} (torch 2.9.1, sglang) ..."
docker build -t "${IMAGE}" "${SCRIPT_DIR}"

echo "Starting SGLang (${MODEL_ID}) on ${HOST_ADDR}:${PORT}"
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -p "${PORT}:${PORT}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e MODEL_ID="${MODEL_ID}" \
  -e HOST="${HOST_ADDR}" \
  -e PORT="${PORT}" \
  -e TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}" \
  -e TOKENIZER_PATH="${TOKENIZER_PATH:-}" \
  -e SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-}" \
  -e SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-}" \
  -e SGLANG_DTYPE="${SGLANG_DTYPE:-}" \
  -e SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-}" \
  -e SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}" \
  "${IMAGE}"
