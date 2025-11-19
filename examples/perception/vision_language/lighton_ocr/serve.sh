#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMG="${CVL_IMAGE:-lighton-ocr}"
MODEL_ID="${LIGHTON_OCR_MODEL_ID:-lightonai/LightOnOCR-1B-1025}"
SERVED_NAME="${LIGHTON_OCR_SERVE_NAME:-lighton-ocr}"
HOST_ADDR="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

CONTAINER_NAME="${CVL_CONTAINER_NAME:-}"

DOCKER_ARGS=(
  --rm
  --gpus=all
  --shm-size 16g
  --ipc=host
  -p "${PORT}:${PORT}"
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface"
  --workdir /workspace
)

if [[ -n "$CONTAINER_NAME" ]]; then
  DOCKER_ARGS+=(--name "$CONTAINER_NAME")
fi

echo "Starting vLLM server for ${MODEL_ID} on ${HOST_ADDR}:${PORT} (served name: ${SERVED_NAME})"

docker run "${DOCKER_ARGS[@]}" "$IMG" bash -lc "
  export HF_HUB_CACHE=/root/.cache/huggingface
  vllm serve \"${MODEL_ID}\" \
    --served-model-name \"${SERVED_NAME}\" \
    --host \"${HOST_ADDR}\" \
    --port \"${PORT}\" \
    --tensor-parallel-size \"${TP_SIZE}\" \
    --limit-mm-per-prompt '{\"image\": 1}' \
    --async-scheduling \
    --trust-remote-code \
    ${LIGHTON_OCR_EXTRA_SERVE_ARGS:-}
"
