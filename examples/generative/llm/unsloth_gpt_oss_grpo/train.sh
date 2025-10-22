#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

set -e

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Will use cached model if available."
    echo "If model download fails, set HF_TOKEN: export HF_TOKEN=your_huggingface_token"
fi

IMAGE_NAME="gpt_oss_grpo"

echo "=== GPT-OSS GRPO Training ==="
echo "Running training in Docker container..."

docker run --runtime nvidia \
    --rm \
    -v $(pwd):/workspace \
    -v $REPO_ROOT/data/container_cache:/root/.cache \
    -e HF_TOKEN=$HF_TOKEN \
    $IMAGE_NAME \
    python3 train.py

echo "✅ Training complete!"
