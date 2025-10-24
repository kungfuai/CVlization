#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Image name
IMG="${CVL_IMAGE:-trl_sft}"

# Optional: Set HuggingFace token for gated models (Llama, etc.)
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Warning: HF_TOKEN not set. Using non-gated model (Qwen 0.5B)."
    echo "To use Llama models, set: export HF_TOKEN=your_huggingface_token"
fi

echo "=== Llama 3B SFT Training with TRL ==="

# Create output directory
mkdir -p "$SCRIPT_DIR/outputs"

# Mount workspace as writable (training writes outputs to /workspace)
docker run --rm --gpus=all --shm-size 16G \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "PYTHONUNBUFFERED=1" \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    "$IMG" "$@"

echo "✅ Training complete!"
