#!/bin/bash

# Quick smoke test with 2 training steps

# Optional: Set HuggingFace token (not required for Qwen)
if [ -z "$HF_TOKEN" ]; then
    echo "Note: HF_TOKEN not set. Using non-gated model (Qwen 0.5B)."
fi

IMAGE_NAME="trl_sft"
CONTAINER_NAME="trl_sft_test"

echo "=== TRL SFT Smoke Test (2 steps) ==="

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
