#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
IMAGE="${LLAMA_CPP_IMAGE:-cvl-llama-cpp}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B-GGUF:Q4_K_M}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

cd "$SCRIPT_DIR"
mkdir -p outputs

echo "Running llama.cpp inference in container (${IMAGE}) with model ${MODEL_ID}"
docker run --rm --gpus all --ipc=host --shm-size 16g \
  ${CUDA_VISIBLE_DEVICES:+-e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES} \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}:/workspace" \
  -v "${REPO_ROOT}:/cvlization_repo:ro" \
  -w /workspace \
  -e PYTHONPATH="/cvlization_repo" \
  -e MODEL_ID="${MODEL_ID}" \
  -e MODEL_PATH="${MODEL_PATH:-}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e LLAMA_CONTEXT_LENGTH="${LLAMA_CONTEXT_LENGTH:-}" \
  -e LLAMA_GPU_LAYERS="${LLAMA_GPU_LAYERS:-}" \
  -e LLAMA_PARALLEL_SLOTS="${LLAMA_PARALLEL_SLOTS:-1}" \
  -e LLAMA_REASONING_FORMAT="${LLAMA_REASONING_FORMAT:-auto}" \
  -e LLAMA_USE_JINJA="${LLAMA_USE_JINJA:-1}" \
  -e LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-1}" \
  ${LLAMA_EXTRA_ARGS:+-e LLAMA_EXTRA_ARGS="$LLAMA_EXTRA_ARGS"} \
  -v "${WORK_DIR}:/mnt/cvl/workspace" \
  -e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
  -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  "${IMAGE}" \
  python3 predict.py "$@"
