#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${LLAMA_CPP_IMAGE:-cvl-llama-cpp}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B-GGUF:Q4_K_M}"
HOST_ADDR="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

echo "Building ${IMAGE} (llama.cpp) ..."
docker build -t "${IMAGE}" "${SCRIPT_DIR}"

echo "Starting llama-server (${MODEL_ID}) on ${HOST_ADDR}:${PORT}"
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -p "${PORT}:${PORT}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e MODEL_ID="${MODEL_ID}" \
  -e MODEL_PATH="${MODEL_PATH:-}" \
  -e HOST="${HOST_ADDR}" \
  -e PORT="${PORT}" \
  -e LLAMA_CONTEXT_LENGTH="${LLAMA_CONTEXT_LENGTH:-}" \
  -e LLAMA_GPU_LAYERS="${LLAMA_GPU_LAYERS:-}" \
  -e LLAMA_PARALLEL_SLOTS="${LLAMA_PARALLEL_SLOTS:-}" \
  -e LLAMA_REASONING_FORMAT="${LLAMA_REASONING_FORMAT:-}" \
  -e LLAMA_USE_JINJA="${LLAMA_USE_JINJA:-}" \
  -e LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-}" \
  -e LLAMA_API_KEY="${LLAMA_API_KEY:-}" \
  -e LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "${IMAGE}"
