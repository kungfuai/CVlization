#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VARIANT="${QWEN3_VL_VARIANT:-2b}"
case "$VARIANT" in
  2b) MODEL_ID="Qwen/Qwen3-VL-2B-Instruct" ;;
  4b) MODEL_ID="Qwen/Qwen3-VL-4B-Instruct" ;;
  8b) MODEL_ID="Qwen/Qwen3-VL-8B-Instruct" ;;
  *)
    echo "Unknown Qwen3-VL variant: $VARIANT" >&2
    exit 1
    ;;
esac

MODEL_NAME="${QWEN3_VL_SERVE_NAME:-qwen3-vl-$VARIANT}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${QWEN3_VL_MAX_MODEL_LEN:-65536}"

echo "Starting vLLM server for $MODEL_ID on $HOST:$PORT (served name: $MODEL_NAME)"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --trust-remote-code \
  ${QWEN3_VL_EXTRA_SERVE_ARGS:-}
