#!/bin/bash

# Quick smoke test for Surya OCR
echo "Running Surya OCR smoke test..."

docker run --runtime nvidia \
    -v $(pwd)/examples/doc_ai/surya:/workspace \
    -v $(pwd)/data/container_cache:/root/.cache \
    surya \
    python3 -c "
import torch
print('Testing Surya OCR installation...')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

try:
    print('Importing Surya modules...')
    from surya.model.detection.model import load_model as load_det_model
    from surya.model.recognition.model import load_model as load_rec_model

    print('Loading detection model...')
    det_model = load_det_model()

    print('Loading recognition model...')
    rec_model = load_rec_model()

    print('✓ All models loaded successfully!')
    print('Smoke test passed!')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"
