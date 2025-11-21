#!/bin/bash
set -e

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get image name from directory
IMAGE_NAME="$(basename "$SCRIPT_DIR")"

# Define cache directory
CACHE_DIR="${HOME}/.cache/cvlization"
mkdir -p "$CACHE_DIR"

# Check if scenes argument provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <scenes.json or comma-separated prompts> [additional options]"
    echo ""
    echo "Examples:"
    echo "  $0 scenes.json"
    echo "  $0 'Wide shot of a forest,Close-up of rustling leaves,Camera pans to reveal a deer'"
    exit 1
fi

SCENES="$1"

echo "Generating multi-scene sequence..."

# If it's a file, mount it
if [ -f "$SCENES" ]; then
    SCENES_ABS="$(cd "$(dirname "$SCENES")" && pwd)/$(basename "$SCENES")"
    echo "Using scenes file: $SCENES"

    docker run --rm --gpus=all \
        -v "$CACHE_DIR:/root/.cache" \
        -v "$SCRIPT_DIR:/workspace" \
        -v "$(dirname "$SCENES_ABS"):/input" \
        --workdir /workspace \
        "$IMAGE_NAME" \
        python3 generate_sequence.py \
            --scenes "/input/$(basename "$SCENES_ABS")" \
            "${@:2}"
else
    # Treat as comma-separated prompts
    echo "Using prompts: $SCENES"

    docker run --rm --gpus=all \
        -v "$CACHE_DIR:/root/.cache" \
        -v "$SCRIPT_DIR:/workspace" \
        --workdir /workspace \
        "$IMAGE_NAME" \
        python3 generate_sequence.py \
            --scenes "$SCENES" \
            "${@:2}"
fi

echo "Sequence generation complete!"
echo "Output saved to: outputs/sequence/"
