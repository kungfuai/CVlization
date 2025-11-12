#!/bin/bash
# Adapter for Qwen3-VL-2B on CheckboxQA
# Usage: ./qwen3_vl_2b.sh <pdf_path> <question> --output <output_file>

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

# Run Qwen3-VL-2B with question as custom prompt
# Follow CVlization pattern: mount repo for cvlization package
QWEN_DIR="$REPO_ROOT/examples/perception/vision_language/qwen3_vl_2b"

docker run --runtime nvidia --rm \
    -v "$QWEN_DIR:/workspace" \
    -v "$IMG_DIR:/inputs:ro" \
    -v "$OUTPUT_DIR:/outputs" \
    -v "$REPO_ROOT:/cvlization_repo:ro" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e PYTHONPATH=/cvlization_repo \
    -e HF_TOKEN=$HF_TOKEN \
    qwen3-vl-2b \
    python3 predict.py \
        --image "/inputs/$IMG_NAME" \
        --output "/outputs/$OUTPUT_NAME" \
        --task vqa \
        --prompt "$QUESTION"

# Cleanup temp image
rm -f "$TMP_IMG"
