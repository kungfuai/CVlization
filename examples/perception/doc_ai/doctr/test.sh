#!/bin/bash

# Quick smoke test for docTR example
echo "Running docTR smoke test..."

docker run --gpus=all \
    -v $(pwd)/examples/perception/doc_ai/doctr:/workspace \
    -v $(pwd)/${CVL_HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
    doctr \
    python3 -c "
import torch
from doctr.models import ocr_predictor

print('Testing docTR model loading...')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Quick test that docTR can load models
try:
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    print('Model loaded successfully!')
except Exception as e:
    print(f'Error loading model: {e}')
    exit(1)

print('Smoke test passed!')
"
