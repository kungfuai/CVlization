#!/bin/bash

# Quick smoke test for dots.ocr example
echo "Running dots.ocr smoke test..."

docker run --gpus=all \
    -v $(pwd)/examples/doc_ai/dots-ocr:/workspace \
    -v $(pwd)/${CVL_HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN \
    dots_ocr \
    python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

print('Testing dots.ocr model loading...')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Quick test that transformers can load the model config
try:
    processor = AutoProcessor.from_pretrained('rednote-hilab/dots.ocr', trust_remote_code=True)
    print('Model processor loaded successfully!')
except Exception as e:
    print(f'Error loading processor: {e}')
    exit(1)

print('Smoke test passed!')
"
