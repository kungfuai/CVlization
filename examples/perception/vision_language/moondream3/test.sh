#!/bin/bash

# Quick smoke test for Moondream3
echo "Running Moondream3 smoke test..."

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set"
    echo "Moondream3 is a gated model that requires authentication."
    echo ""
    echo "Please set your HuggingFace token:"
    echo "  export HF_TOKEN=your_huggingface_token"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Request access to the model: https://huggingface.co/moondream/moondream3-preview"
    exit 1
fi

docker run --gpus=all \
    -v $(pwd)/examples/doc_ai/moondream3:/workspace \
    -v $(pwd)/data/container_cache:/root/.cache \
    -e HF_TOKEN=$HF_TOKEN \
    moondream3 \
    python3 -c "
import torch
from transformers import AutoModelForCausalLM

print('Testing Moondream3 model loading...')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

try:
    print('Loading model (this may take a moment)...')
    model = AutoModelForCausalLM.from_pretrained(
        'moondream/moondream3-preview',
        trust_remote_code=True,
        device_map={'': 'cuda'}
    )
    print('Model loaded successfully!')
    print(f'Model device: {next(model.parameters()).device}')
    print('Smoke test passed!')
except Exception as e:
    print(f'Error loading model: {e}')
    exit(1)
"
