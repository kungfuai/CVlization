#!/usr/bin/env bash
# Smoke-test the MolmoAct2 inference Docker image (no GPU model load)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up 3 levels: molmoact2 > physical > examples > repo root
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
IMAGE_NAME="${CVL_IMAGE:-molmoact2-inference:latest}"

echo "Testing MolmoAct2 inference Docker image..."
echo "Image: ${IMAGE_NAME}"
echo ""

# Common flags used for all tests (mirrors predict.sh mounts)
COMMON_FLAGS=(
    --rm
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace"
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly"
    --env "PYTHONPATH=/cvlization_repo"
    --workdir /workspace
)

# Test 1: Python and PyTorch
echo "Test 1: Python and PyTorch version check..."
docker run "${COMMON_FLAGS[@]}" "${IMAGE_NAME}" python -c "
import sys, torch
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print('PASSED')
"

# Test 2: Transformers classes
echo ""
echo "Test 2: Transformers import check..."
docker run "${COMMON_FLAGS[@]}" "${IMAGE_NAME}" python -c "
from transformers import AutoModelForImageTextToText, AutoProcessor
print('PASSED: AutoModelForImageTextToText and AutoProcessor available')
"

# Test 3: Other dependencies
echo ""
echo "Test 3: Dependency imports..."
docker run "${COMMON_FLAGS[@]}" "${IMAGE_NAME}" python -c "
import numpy, PIL, scipy, huggingface_hub
print('PASSED: numpy, Pillow, scipy, huggingface_hub')
"

# Test 4: cvlization.paths available via PYTHONPATH
echo ""
echo "Test 4: cvlization.paths import (via PYTHONPATH mount)..."
docker run "${COMMON_FLAGS[@]}" "${IMAGE_NAME}" python -c "
from cvlization.paths import resolve_input_path, resolve_output_path
print('PASSED: cvlization.paths available')
"

# Test 5: predict.py syntax check
echo ""
echo "Test 5: predict.py syntax..."
docker run "${COMMON_FLAGS[@]}" "${IMAGE_NAME}" python -m py_compile predict.py
echo "PASSED"

# Test 6: Help output
echo ""
echo "Test 6: Help output..."
docker run "${COMMON_FLAGS[@]}" "${IMAGE_NAME}" python predict.py --help

echo ""
echo "=========================================="
echo "All smoke tests passed!"
echo ""
echo "Full inference requires GPU and downloads ~20GB model on first run."
echo "Run: ./predict.sh"
