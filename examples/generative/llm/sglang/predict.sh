#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
IMAGE="${SGLANG_IMAGE:-cvl-sglang}"
MODEL_ID="${MODEL_ID:-allenai/Olmo-3-7B-Instruct}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
# Find repo root for cvlization package
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

cd "$SCRIPT_DIR"
mkdir -p outputs

echo "Running SGLang inference in container (${IMAGE}) with model ${MODEL_ID}"
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}:/workspace" \
  -w /workspace \
  -e MODEL_ID="${MODEL_ID}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}" \
  -e SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}" \
  -e SGLANG_DTYPE="${SGLANG_DTYPE:-bfloat16}" \
  -e SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.9}" \
  -e TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}" \
  -v "${WORK_DIR}:/mnt/cvl/workspace" \
  -e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
  -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  "${IMAGE}" \
  python3 predict.py "$@"
