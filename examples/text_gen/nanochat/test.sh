#!/bin/bash
# Quick test to verify nanochat container works

echo "Testing nanochat Docker setup..."
echo ""

docker run --rm nanochat bash -c "
    cd nanochat
    echo '=== Python & PyTorch ==='
    uv run python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'

    echo ''
    echo '=== nanochat module ==='
    uv run python -c 'import nanochat; print(\"nanochat imported successfully\")'

    echo ''
    echo '=== Installed packages (sample) ==='
    uv pip list | grep -E 'torch|nanochat|wandb|tiktoken' || true
"

echo ""
echo "Test complete! nanochat container is ready."
