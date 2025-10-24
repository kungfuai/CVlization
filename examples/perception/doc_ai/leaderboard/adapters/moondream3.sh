#!/bin/bash
# Adapter for moondream3 - normalizes interface
# Usage: ./moondream3.sh <input_image> --output <output_file> [args]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_IMAGE="$1"
shift

# Get absolute paths
INPUT_ABS=$(realpath "$INPUT_IMAGE")
INPUT_DIR=$(dirname "$INPUT_ABS")
INPUT_NAME=$(basename "$INPUT_ABS")

# Parse output flag
OUTPUT_FILE=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Set default output
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="outputs/result.txt"
fi
OUTPUT_ABS=$(realpath -m "$OUTPUT_FILE")
OUTPUT_DIR=$(dirname "$OUTPUT_ABS")
OUTPUT_NAME=$(basename "$OUTPUT_ABS")

mkdir -p "$OUTPUT_DIR"

# Run docker directly with proper mounts
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
cd "$SCRIPT_DIR/../../moondream3"

docker run --gpus=all --rm \
    -v "$(pwd):/workspace" \
    -v "$INPUT_DIR:/inputs:ro" \
    -v "$OUTPUT_DIR:/outputs" \
    -v "$REPO_ROOT/data/container_cache:/root/.cache" \
    -e HF_TOKEN=$HF_TOKEN \
    moondream3 \
    python3 predict.py --image "/inputs/$INPUT_NAME" --output "/outputs/$OUTPUT_NAME" "${ARGS[@]}"
