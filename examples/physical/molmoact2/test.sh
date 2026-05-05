#!/usr/bin/env bash
# Smoke-test the MolmoAct2 inference Docker image (no GPU model load)
set -euo pipefail

IMAGE_NAME="${CVL_IMAGE:-molmoact2-inference:latest}"

echo "Testing MolmoAct2 inference Docker image..."
echo "Image: ${IMAGE_NAME}"
echo ""

# Test 1: Python and PyTorch
echo "Test 1: Python and PyTorch version check..."
docker run --rm "${IMAGE_NAME}" python -c "
import sys, torch
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print('PASSED')
"

# Test 2: Transformers + trust_remote_code classes
echo ""
echo "Test 2: Transformers import check..."
docker run --rm "${IMAGE_NAME}" python -c "
from transformers import AutoModelForImageTextToText, AutoProcessor
print('PASSED: AutoModelForImageTextToText and AutoProcessor available')
"

# Test 3: Other dependencies
echo ""
echo "Test 3: Dependency imports..."
docker run --rm "${IMAGE_NAME}" python -c "
import numpy, PIL, scipy, huggingface_hub
print('PASSED: numpy, Pillow, scipy, huggingface_hub')
"

# Test 4: Inference script syntax
echo ""
echo "Test 4: Inference script syntax..."
docker run --rm "${IMAGE_NAME}" python -m py_compile inference.py
echo "PASSED"

# Test 5: Help output
echo ""
echo "Test 5: Help output..."
docker run --rm "${IMAGE_NAME}" python inference.py --help

echo ""
echo "=========================================="
echo "All smoke tests passed!"
echo ""
echo "Full inference requires GPU and downloads ~20GB model."
echo "Run: ./predict.sh"
