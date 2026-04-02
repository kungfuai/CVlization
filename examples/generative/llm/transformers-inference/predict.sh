#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
IMAGE="${TRANSFORMERS_IMAGE:-cvl-transformers-inference}"
MODEL_ID="${MODEL_ID:-allenai/Olmo-Hybrid-Instruct-DPO-7B}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

cd "$SCRIPT_DIR"
mkdir -p outputs

echo "Running transformers inference in container (${IMAGE}) with model ${MODEL_ID}"
docker run --rm --gpus all --ipc=host --shm-size 16g \
  ${CUDA_VISIBLE_DEVICES:+-e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES} \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}:/workspace" \
  -v "${REPO_ROOT}:/cvlization_repo:ro" \
  -w /workspace \
  -e PYTHONPATH="/cvlization_repo" \
  -e MODEL_ID="${MODEL_ID}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e DTYPE="${DTYPE:-bfloat16}" \
  ${IMAGE_PATH:+-e IMAGE_PATH="$IMAGE_PATH"} \
  -v "${WORK_DIR}:/mnt/cvl/workspace" \
  -e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
  -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  "${IMAGE}" \
  python3 predict.py "$@"
