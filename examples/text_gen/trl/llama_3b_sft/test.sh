#!/bin/bash

# Quick smoke test with 2 training steps

# Optional: Set HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Required for Llama models."
    echo "Set it with: export HF_TOKEN=your_huggingface_token"
    exit 1
fi

IMAGE_NAME="llama_3b_trl_sft"
CONTAINER_NAME="llama_3b_trl_sft_test"

echo "=== Llama 3B SFT Smoke Test (2 steps) ==="

# Get absolute path to repo root
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="$REPO_ROOT/data/container_cache"

# Create cache directory
mkdir -p "$CACHE_DIR/huggingface"

# Create test config with reduced steps
cat config.yaml | sed 's/max_steps: 20/max_steps: 2/' > config_test.yaml

# Run smoke test
docker run --rm --gpus all \
    --name "$CONTAINER_NAME" \
    -v "$(pwd)":/workspace \
    -v "$CACHE_DIR/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN="$HF_TOKEN" \
    "$IMAGE_NAME" \
    bash -c "cp config_test.yaml config.yaml && python3 train.py"

# Cleanup
rm -f config_test.yaml

echo "✅ Smoke test complete!"
