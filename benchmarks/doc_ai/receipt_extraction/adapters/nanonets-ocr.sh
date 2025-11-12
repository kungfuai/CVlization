#!/bin/bash
# Adapter for nanonets-ocr
# Usage: ./nanonets-ocr.sh <input_image> --output <output_file> [args]

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
REPO_ROOT="$SCRIPT_DIR/../../../.."
cd "$REPO_ROOT/examples/doc_ai/nanonets-ocr"

docker run --rm --gpus all \
    -v "$INPUT_DIR:/app/inputs:ro" \
    -v "$OUTPUT_DIR:/app/outputs" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    nanonets-ocr \
    python predict.py "/app/inputs/$INPUT_NAME" --output "/app/outputs/$OUTPUT_NAME" --device cuda "${ARGS[@]}"
