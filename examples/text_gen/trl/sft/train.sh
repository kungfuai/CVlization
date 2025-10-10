#!/bin/bash

# Optional: Set HuggingFace token for gated models (Llama, etc.)
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Using non-gated model (Qwen 0.5B)."
    echo "To use Llama models, set: export HF_TOKEN=your_huggingface_token"
fi

IMAGE_NAME="trl_sft"
CONTAINER_NAME="trl_sft_training"

echo "=== Llama 3B SFT Training with TRL ==="
echo "Running training in Docker container..."

# Get absolute path to repo root (3 levels up from this script)
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="$REPO_ROOT/data/container_cache"

# Create cache directory if it doesn't exist
mkdir -p "$CACHE_DIR/huggingface"

# Run training container
docker run --rm --gpus all \
    --name "$CONTAINER_NAME" \
    -v "$(pwd)":/workspace \
    -v "$CACHE_DIR/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN="$HF_TOKEN" \
    "$IMAGE_NAME"

echo "✅ Training complete!"
