#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${VLLM_IMAGE:-cvl-vllm}"
MODEL_ID="${MODEL_ID:-allenai/Olmo-3-7B-Instruct}"
HOST_ADDR="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Building ${IMAGE} (torch 2.9.1, vLLM) ..."
docker build -t "${IMAGE}" "${SCRIPT_DIR}"

echo "Starting vLLM (${MODEL_ID}) on ${HOST_ADDR}:${PORT}"
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -p "${PORT}:${PORT}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e MODEL_ID="${MODEL_ID}" \
  -e HOST="${HOST_ADDR}" \
  -e PORT="${PORT}" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}" \
  -e TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}" \
  -e VLLM_TP_SIZE="${VLLM_TP_SIZE:-}" \
  -e VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}" \
  -e VLLM_DTYPE="${VLLM_DTYPE:-}" \
  -e VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}" \
  -e VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}" \
  "${IMAGE}"
