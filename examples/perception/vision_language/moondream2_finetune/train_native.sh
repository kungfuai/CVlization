#!/bin/bash
set -e

echo "Starting Moondream2 fine-tuning (Native Implementation)..."
echo ""

# Run Docker container with GPU support, passing all arguments through
docker run --rm \
    --gpus all \
    -v "$(pwd):/workspace" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    moondream2-finetune \
    python train_native.py "$@"

echo ""
echo "✓ Training complete!"
