#!/bin/bash

# Quick smoke test with 2 training steps

# Optional: Set HuggingFace token (not required for Qwen)
if [ -z "$HF_TOKEN" ]; then
    echo "Note: HF_TOKEN not set. Using non-gated model (Qwen 0.5B)."
fi

IMAGE_NAME="trl_sft"
CONTAINER_NAME="trl_sft_test"

echo "=== TRL SFT Smoke Test (2 steps) ==="

# Use configurable HuggingFace cache (defaults to $HOME/.cache/huggingface)
CACHE_DIR="${CVL_HF_CACHE:-$HOME/.cache/huggingface}"

# Create cache directory if needed
mkdir -p "$CACHE_DIR"

# Create test config with reduced steps
cat config.yaml | sed 's/max_steps: 20/max_steps: 2/' > config_test.yaml

# Run smoke test
docker run --rm --gpus all \
    --name "$CONTAINER_NAME" \
    -v "$(pwd)":/workspace \
    -v "$CACHE_DIR:/root/.cache/huggingface" \
    -e HF_TOKEN="$HF_TOKEN" \
    "$IMAGE_NAME" \
    bash -c "cp config_test.yaml config.yaml && python3 train.py"

# Cleanup
rm -f config_test.yaml

echo "✅ Smoke test complete!"
