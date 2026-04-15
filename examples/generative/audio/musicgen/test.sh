#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-musicgen}"
OUTPUT="${SCRIPT_DIR}/outputs/test_musicgen.wav"

mkdir -p "${HOME}/.cache/huggingface"
mkdir -p "${HOME}/.cache/torch"
mkdir -p "${SCRIPT_DIR}/outputs"

echo "=== MusicGen Smoke Test ==="

docker run --rm --gpus=all \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${HOME}/.cache/torch,dst=/root/.cache/torch" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=/root/.cache/huggingface" \
  --env "TORCH_HOME=/root/.cache/torch" \
  "$IMG" python predict.py --prepare-sample-data

docker run --rm --gpus=all \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${HOME}/.cache/torch,dst=/root/.cache/torch" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=/root/.cache/huggingface" \
  --env "TORCH_HOME=/root/.cache/torch" \
  "$IMG" python predict.py \
    --model debug \
    --text "short upbeat electronic music loop" \
    --duration 1 \
    --seed 123 \
    --output test_musicgen.wav \
    --no-progress

echo "=== Verifying output ==="
if [ -f "$OUTPUT" ]; then
  SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null)
  echo "Output file: outputs/test_musicgen.wav"
  echo "File size: ${SIZE} bytes"
  if [ "$SIZE" -gt 1000 ]; then
    echo "Test PASSED: Audio file generated successfully"
  else
    echo "Test FAILED: Audio file too small"
    exit 1
  fi
else
  echo "Test FAILED: Output file not created"
  exit 1
fi
