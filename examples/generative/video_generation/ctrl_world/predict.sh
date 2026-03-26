#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"

IMG="${CVL_IMAGE:-cvlization/ctrl-world:latest}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Running Ctrl-World replay inference in container (${IMG})"
docker run --rm --gpus all \
  --shm-size 16g \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}:/workspace" \
  -v "${REPO_ROOT}:/cvlization_repo:ro" \
  -v "${WORK_DIR}:/mnt/cvl/workspace" \
  -e PYTHONPATH="/cvlization_repo" \
  -e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
  -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  ${CUDA_VISIBLE_DEVICES:+-e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"} \
  -w /workspace \
  "${IMG}" \
  python predict.py "$@"
