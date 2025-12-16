#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${VLLM_IMAGE:-cvl-vllm}"
MODEL_ID="${MODEL_ID:-allenai/Olmo-3-7B-Instruct}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

cd "$SCRIPT_DIR"
mkdir -p outputs

echo "Running local vLLM inference in container (${IMAGE}) with model ${MODEL_ID}"
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}:/workspace" \
  -w /workspace \
  -e MODEL_ID="${MODEL_ID}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e VLLM_MODE="${VLLM_MODE:-chat}" \
  -e VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}" \
  -e VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}" \
  -e VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}" \
  -e VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
  -e VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}" \
  "${IMAGE}" \
  python3 predict.py "$@"
