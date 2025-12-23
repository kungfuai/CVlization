#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
IMAGE="${CLARA_IMAGE:-cvl-clara}"
MODEL_ID="${MODEL_ID:-apple/CLaRa-7B-Instruct}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
# Find repo root for cvlization package
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

cd "${SCRIPT_DIR}"
mkdir -p outputs

echo "Running CLaRa inference in container (${IMAGE}) with model ${MODEL_ID}"
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}:/workspace" \
  -w /workspace \
  -e MODEL_ID="${MODEL_ID}" \
  -e REVISION="${REVISION:-}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${WORK_DIR}:/mnt/cvl/workspace" \
  -e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
    -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  "${IMAGE}" \
  python3 predict.py "$@"
