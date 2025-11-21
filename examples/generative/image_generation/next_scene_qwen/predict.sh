#!/bin/bash
set -e

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get image name from directory
IMAGE_NAME="$(basename "$SCRIPT_DIR")"

# Define cache directory
CACHE_DIR="${HOME}/.cache/cvlization"
mkdir -p "$CACHE_DIR"

# Default prompt
PROMPT="${1:-The camera moves slightly forward as sunlight breaks through clouds}"

echo "Generating next scene..."
echo "Prompt: Next Scene: $PROMPT"

docker run --rm --gpus=all \
    -v "$CACHE_DIR:/root/.cache" \
    -v "$SCRIPT_DIR:/workspace" \
    --workdir /workspace \
    "$IMAGE_NAME" \
    python3 generate_scene.py \
        --prompt "$PROMPT" \
        "${@:2}"

echo "Scene generation complete!"
echo "Output saved to: outputs/scene.png"
