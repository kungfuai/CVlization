#!/bin/bash
# Adapter for docling-serve (plain Docling library)
# Usage: ./docling-serve.sh <input_image> --output <output_file> [args]

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

if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="outputs/result.txt"
fi
OUTPUT_ABS=$(realpath -m "$OUTPUT_FILE")
OUTPUT_DIR=$(dirname "$OUTPUT_ABS")
OUTPUT_NAME=$(basename "$OUTPUT_ABS")

mkdir -p "$OUTPUT_DIR"

# Run docker with proper mounts
cd "$SCRIPT_DIR/../../docling-serve"

docker run --rm \
    -v "$INPUT_DIR:/app/inputs:ro" \
    -v "$OUTPUT_DIR:/app/outputs" \
    docling-serve \
    python predict.py "/app/inputs/$INPUT_NAME" --output "/app/outputs/$OUTPUT_NAME" "${ARGS[@]}"
