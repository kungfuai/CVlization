#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-voxcpm}"

echo "=== VoxCPM1.5 Smoke Test ==="
echo ""

# Run inference with default text
docker run --rm --gpus=all \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=/root/.cache/huggingface" \
  "$IMG" python predict.py \
    --text "Hello! This is a test of VoxCPM text to speech." \
    --output test_output.wav

echo ""
echo "=== Verifying output ==="

if [ -f "${SCRIPT_DIR}/outputs/test_output.wav" ]; then
    SIZE=$(stat -f%z "${SCRIPT_DIR}/outputs/test_output.wav" 2>/dev/null || stat -c%s "${SCRIPT_DIR}/outputs/test_output.wav" 2>/dev/null)
    echo "Output file: outputs/test_output.wav"
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
