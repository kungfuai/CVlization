#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
IMAGE="${LLAMA_CPP_IMAGE:-cvl-llama-cpp}"
MODEL_ID="${MODEL_ID:-mradermacher/LLaDA-8B-Instruct-GGUF:Q4_K_M}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
# We borrow gpu_utils.py from the sibling llama_cpp example.
SIBLING_DIR="$(cd "${SCRIPT_DIR}/../llama_cpp" && pwd)"

cd "$SCRIPT_DIR"
mkdir -p outputs

echo "Running llama.cpp dLLM inference (${IMAGE}) with model ${MODEL_ID}"
docker run --rm --gpus all --ipc=host --shm-size 16g \
  ${CUDA_VISIBLE_DEVICES:+-e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES} \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}:/workspace" \
  -v "${SIBLING_DIR}:/sibling:ro" \
  -v "${REPO_ROOT}:/cvlization_repo:ro" \
  -w /workspace \
  -e PYTHONPATH="/cvlization_repo:/sibling" \
  -e MODEL_ID="${MODEL_ID}" \
  -e MODEL_PATH="${MODEL_PATH:-}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e LLAMA_GPU_LAYERS="${LLAMA_GPU_LAYERS:-}" \
  -e LLAMA_CONTEXT_LENGTH="${LLAMA_CONTEXT_LENGTH:-}" \
  -e LLAMA_N_PREDICT="${LLAMA_N_PREDICT:-}" \
  -e LLAMA_DIFFUSION_STEPS="${LLAMA_DIFFUSION_STEPS:-}" \
  -e LLAMA_DIFFUSION_ALGORITHM="${LLAMA_DIFFUSION_ALGORITHM:-}" \
  -e LLAMA_DIFFUSION_BLOCK_LENGTH="${LLAMA_DIFFUSION_BLOCK_LENGTH:-}" \
  -e LLAMA_DIFFUSION_EPS="${LLAMA_DIFFUSION_EPS:-}" \
  -e LLAMA_DIFFUSION_CFG_SCALE="${LLAMA_DIFFUSION_CFG_SCALE:-}" \
  ${LLAMA_EXTRA_ARGS:+-e LLAMA_EXTRA_ARGS="$LLAMA_EXTRA_ARGS"} \
  -v "${WORK_DIR}:/mnt/cvl/workspace" \
  -e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
  -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  "${IMAGE}" \
  python3 predict.py "$@"
