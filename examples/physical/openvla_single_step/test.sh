#!/usr/bin/env bash
# Test the OpenVLA inference Docker image
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-openvla-inference:latest}"

echo "Testing OpenVLA inference Docker image..."
echo "Image: ${IMAGE_NAME}"
echo ""

# Test 1: Check Python and PyTorch
echo "Test 1: Python and PyTorch version check..."
docker run --rm "${IMAGE_NAME}" python -c "
import sys
import torch
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print('PASSED: PyTorch check')
"

# Test 2: Check transformers and model loading capability
echo ""
echo "Test 2: Transformers import check..."
docker run --rm "${IMAGE_NAME}" python -c "
from transformers import AutoModelForVision2Seq, AutoProcessor
print('PASSED: Transformers AutoClasses available')
"

# Test 3: Check visualization dependencies
echo ""
echo "Test 3: Visualization dependencies..."
docker run --rm "${IMAGE_NAME}" python -c "
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
print('PASSED: Visualization dependencies')
"

# Test 4: Check inference script syntax
echo ""
echo "Test 4: Inference script syntax check..."
docker run --rm "${IMAGE_NAME}" python -m py_compile inference.py
echo "PASSED: Inference script syntax"

# Test 5: Check help output
echo ""
echo "Test 5: Help output..."
docker run --rm "${IMAGE_NAME}" python inference.py --help

echo ""
echo "=========================================="
echo "All tests passed!"
echo ""
echo "Note: Full inference test requires GPU and downloads ~14GB model."
echo "Run './run.sh --interactive' to test with the actual model."
