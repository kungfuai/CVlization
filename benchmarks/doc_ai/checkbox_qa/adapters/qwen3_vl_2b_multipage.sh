#!/bin/bash
# Multi-page adapter for Qwen3-VL-2B with full PDF support

set -e

PDF_PATH="$1"
QUESTION="$2"
OUTPUT_PATH="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Load centralized prompt templates
source "$SCRIPT_DIR/../prompts.sh"
ENHANCED_PROMPT="${ACTIVE_PROMPT}${QUESTION}"

# Convert PDF to absolute path
PDF_ABS=$(realpath "$PDF_PATH")

# Output directory
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
OUTPUT_NAME=$(basename "$OUTPUT_PATH")

mkdir -p "$OUTPUT_DIR"

# Get document name and create temp directory
DOC_NAME=$(basename "$PDF_PATH" .pdf)
TMP_DIR="/tmp/checkbox_qa_multipage_${DOC_NAME}_$$"
mkdir -p "$TMP_DIR"

echo "Processing PDF: $DOC_NAME"

# Get page count
PAGE_COUNT=$(pdfinfo "$PDF_ABS" | grep Pages | awk '{print $2}')
echo "Document has $PAGE_COUNT pages"

# Convert all pages to separate PNG files
pdftoppm -png "$PDF_ABS" "$TMP_DIR/page"

# Get list of generated page files (sorted)
PAGE_FILES=($(ls -1v "$TMP_DIR"/page-*.png))

echo "Converted ${#PAGE_FILES[@]} pages to images"

# Build --images argument with all page paths (using container paths)
IMAGES_ARG=""
for page_file in "${PAGE_FILES[@]}"; do
    # Use /inputs/ path as it will be inside the container
    IMAGES_ARG="$IMAGES_ARG /inputs/$(basename "$page_file")"
done

echo "Passing ${#PAGE_FILES[@]} images to VLM"

# Run Qwen3-VL-2B with multi-image support
QWEN_DIR="$REPO_ROOT/examples/perception/vision_language/qwen3_vl_2b"

docker run --runtime nvidia --rm \
    -v "$QWEN_DIR:/workspace" \
    -v "$TMP_DIR:/inputs:ro" \
    -v "$OUTPUT_DIR:/outputs" \
    -v "$REPO_ROOT:/cvlization_repo:ro" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e PYTHONPATH=/cvlization_repo \
    -e HF_TOKEN=$HF_TOKEN \
    -e PROMPT="$ENHANCED_PROMPT" \
    qwen3-vl-2b \
    bash -c "python3 predict.py --images $IMAGES_ARG --output /outputs/$OUTPUT_NAME --task vqa --prompt \"\$PROMPT\""

# Cleanup temporary files
rm -rf "$TMP_DIR"

echo "Multi-page processing complete"
