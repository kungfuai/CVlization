#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="$(basename "$SCRIPT_DIR")"
CACHE_DIR="${HOME}/.cache/cvlization"
mkdir -p "$CACHE_DIR"

# Default values
PROMPT="${PROMPT:-A person walking through a forest at sunset}"
OUTPUT="${OUTPUT:-outputs/video.mp4}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
SEED="${SEED:-42}"
FPS="${FPS:-24}"

echo "Krea Realtime Text-to-Video Generation"
echo "========================================"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Blocks: $NUM_BLOCKS"
echo ""

docker run --rm --gpus=all \
    -v "$CACHE_DIR:/root/.cache" \
    -v "$SCRIPT_DIR:/workspace" \
    --workdir /workspace \
    "$IMAGE_NAME" \
    python3 predict.py \
        --prompt "$PROMPT" \
        --output "$OUTPUT" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --num-blocks "$NUM_BLOCKS" \
        --seed "$SEED" \
        --fps "$FPS"
