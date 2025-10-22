#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root and set cache directory
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
CACHE_DIR="$REPO_ROOT/data/container_cache"

# Default values (relative to script directory)
IMAGE_PATH="examples/sample.jpg"
OUTPUT_PATH="outputs/result.txt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Run from script directory, works from anywhere
docker run --runtime nvidia \
    -v "$SCRIPT_DIR":/workspace \
    -v "$CACHE_DIR":/root/.cache \
    -e HF_TOKEN=$HF_TOKEN \
    moondream2 \
    python3 predict.py --image "$IMAGE_PATH" --output "$OUTPUT_PATH" "$@"
