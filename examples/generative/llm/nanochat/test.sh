#!/bin/bash
# Quick test to verify nanochat container works

echo "Testing nanochat Docker setup..."
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm \
    --workdir /workspace \
    -v "$SCRIPT_DIR:/workspace" \
    nanochat bash -c "
    cd /workspace/nanochat
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
