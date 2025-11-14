#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_ID="${PHI4_MODEL_ID:-microsoft/Phi-4-multimodal-instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SERVED_NAME="${PHI4_SERVED_NAME:-phi-4-multimodal}"
MAX_LEN="${PHI4_MAX_MODEL_LEN:-65536}"

echo "Starting vLLM server for $MODEL_ID on $HOST:$PORT (served name: $SERVED_NAME)"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --trust-remote-code \
  ${PHI4_EXTRA_SERVE_ARGS:-}
