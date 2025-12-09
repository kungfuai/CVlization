#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running MolmoE-1B smoke test..."

docker run --gpus=all \
    -v "$SCRIPT_DIR":/workspace \
    -v "${CVL_HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    molmoe_1b \
    python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

print('Testing MolmoE-1B model loading...')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

try:
    print('Loading model (this may take a moment)...')
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/MolmoE-1B-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    print('Model loaded successfully!')
    print(f'Model class: {model.__class__.__name__}')
    print('Smoke test passed!')
except Exception as e:
    print(f'Error loading model: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"
