#!/bin/bash

# Quick smoke test for Moondream2
echo "Running Moondream2 smoke test..."

docker run --runtime nvidia \
    -v $(pwd)/examples/doc_ai/moondream2:/workspace \
    -v $(pwd)/data/container_cache:/root/.cache \
    -e HF_TOKEN=$HF_TOKEN \
    moondream2 \
    python3 -c "
import torch
from transformers import AutoModelForCausalLM

print('Testing Moondream2 model loading...')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

try:
    print('Loading model (this may take a moment)...')
    model = AutoModelForCausalLM.from_pretrained(
        'vikhyatk/moondream2',
        revision='2025-06-21',
        trust_remote_code=True,
        device_map={'': 'cuda'}
    )
    print('Model loaded successfully!')
    print(f'Model device: {model.device}')
    print('Smoke test passed!')
except Exception as e:
    print(f'Error loading model: {e}')
    exit(1)
"
