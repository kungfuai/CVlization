#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Load .env from main repo if it exists
MAIN_REPO="${HOME}/projects/CVlization"
if [ -f "${MAIN_REPO}/.env" ]; then
    export $(grep -v '^#' "${MAIN_REPO}/.env" | xargs)
fi

# Image name
IMG="${CVL_IMAGE:-cvlization/gemma3-vision:latest}"

# HuggingFace cache directory
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Running Gemma-3 Vision smoke test..."
echo "This will run inference on a sample image to verify the setup works."
echo ""

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Warning: HF_TOKEN not set. Gemma-3 is a gated model - you may need to:"
    echo "  1. Accept the license at https://huggingface.co/google/gemma-3-4b-it"
    echo "  2. Set HF_TOKEN environment variable or run 'huggingface-cli login'"
    echo ""
fi

# Test image URL
TEST_IMAGE="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

docker run --rm --gpus=all \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=/root/.cache/huggingface" \
  ${HF_TOKEN:+--env "HF_TOKEN=${HF_TOKEN}"} \
  "$IMG" python3 predict.py \
    --image "$TEST_IMAGE" \
    --task describe \
    --max-tokens 256

echo ""
echo "✓ Smoke test passed!"
