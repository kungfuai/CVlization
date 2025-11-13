#!/bin/bash
# Adapter for Florence-2 (Base variant) on CheckboxQA
# Usage: ./florence_2.sh <pdf_path> <question> --output <output_file>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDF_PATH="$1"
QUESTION="$2"
shift 2

# Parse output flag
OUTPUT_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="output.txt"
fi

# Get absolute paths
REPO_ROOT="$SCRIPT_DIR/../../../.."
PDF_ABS=$(realpath "$PDF_PATH")
OUTPUT_ABS=$(realpath -m "$OUTPUT_FILE")
OUTPUT_DIR=$(dirname "$OUTPUT_ABS")
OUTPUT_NAME=$(basename "$OUTPUT_ABS")

mkdir -p "$OUTPUT_DIR"

# Convert PDF first page to image
TMP_IMG="/tmp/checkbox_qa_$(basename "$PDF_PATH" .pdf).png"
pdftoppm -png -f 1 -l 1 -singlefile "$PDF_ABS" "${TMP_IMG%.png}"

# Get image paths
IMG_DIR=$(dirname "$TMP_IMG")
IMG_NAME=$(basename "$TMP_IMG")

# Run Florence-2 (base) with detailed caption + question
# Florence-2 uses special task prompts like <MORE_DETAILED_CAPTION>
# We'll use MORE_DETAILED_CAPTION to get detailed understanding
# and append the question
FLORENCE_DIR="$REPO_ROOT/examples/perception/vision_language/florence_2"

docker run --runtime nvidia --rm \
    -v "$FLORENCE_DIR:/workspace" \
    -v "$IMG_DIR:/inputs:ro" \
    -v "$OUTPUT_DIR:/outputs" \
    -v "$REPO_ROOT:/cvlization_repo:ro" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e PYTHONPATH=/cvlization_repo \
    -e HF_TOKEN=$HF_TOKEN \
    florence-2 \
    python3 predict.py \
        --variant base \
        --image "/inputs/$IMG_NAME" \
        --output "/outputs/$OUTPUT_NAME" \
        --task more_detailed_caption \
        --text-input "$QUESTION"

# Cleanup temp image
rm -f "$TMP_IMG"
