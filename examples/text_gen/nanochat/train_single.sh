#!/bin/bash
# Train a small nanochat model on a single GPU
# This is a simplified version that fits on most GPUs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create output directories
mkdir -p "$SCRIPT_DIR/data"
mkdir -p "$SCRIPT_DIR/outputs"

echo "Training nanochat on single GPU..."
echo "Model: depth=8 (tiny), ~200M parameters"
echo ""

docker run --rm --gpus all \
    -v "$SCRIPT_DIR/data:/workspace/data" \
    -v "$SCRIPT_DIR/outputs:/workspace/outputs" \
    -v "$HOME/.cache/nanochat:/root/.cache/nanochat" \
    nanochat \
    bash -c "
        cd /workspace/nanochat

        # Build rustbpe tokenizer
        echo '=== Building tokenizer ==='
        uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

        # Download small dataset (10 shards for tokenizer training)
        echo ''
        echo '=== Downloading training data ==='
        uv run python -m nanochat.dataset -n 10

        # Train tokenizer on first 2 shards (~500M chars)
        echo ''
        echo '=== Training tokenizer ==='
        uv run python -m scripts.tok_train --max_chars=500000000

        # Train tiny model (limited iterations for quick test)
        echo ''
        echo '=== Starting model training ==='
        uv run python -m scripts.base_train --depth=8 --device_batch_size=4 --num_iterations=100

        echo ''
        echo 'Training complete!'
        echo 'Checkpoints saved in /workspace/outputs'
    "

echo ""
echo "Training finished! Check outputs/ directory for checkpoints."
