#!/bin/bash
# Adapter for moondream2 - normalizes interface
# Usage: ./moondream2.sh <input_image> --output <output_file> [args]

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
MOONDREAM2_DIR="$REPO_ROOT/examples/perception/vision_language/moondream2"

# Follow CVlization pattern: mount repo for cvlization package, set PYTHONPATH
docker run --runtime nvidia --rm \
    -v "$MOONDREAM2_DIR:/workspace" \
    -v "$INPUT_DIR:/inputs:ro" \
    -v "$OUTPUT_DIR:/outputs" \
    -v "$REPO_ROOT:/cvlization_repo:ro" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e PYTHONPATH=/cvlization_repo \
    -e HF_TOKEN=$HF_TOKEN \
    moondream2 \
    python3 predict.py --image "/inputs/$INPUT_NAME" --output "/outputs/$OUTPUT_NAME" "${ARGS[@]}"
