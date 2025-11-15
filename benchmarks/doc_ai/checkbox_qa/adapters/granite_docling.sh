#!/bin/bash
# Adapter for Granite-Docling on CheckboxQA (single page, first page only)
# Usage: ./granite_docling.sh <pdf_path> <question> --output <output_file>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDF_PATH="$1"
QUESTION="$2"
shift 2

# Parse output flag
OUTPUT_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
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

REPO_ROOT="$SCRIPT_DIR/../../../.."
PDF_ABS=$(realpath "$PDF_PATH")
OUTPUT_ABS=$(realpath -m "$OUTPUT_FILE")
OUTPUT_DIR=$(dirname "$OUTPUT_ABS")
OUTPUT_NAME=$(basename "$OUTPUT_ABS")
mkdir -p "$OUTPUT_DIR"

DOC_ID="$(basename "$PDF_PATH" .pdf)"
PAGE_CACHE="${CHECKBOX_QA_PAGE_CACHE:-$REPO_ROOT/benchmarks/doc_ai/checkbox_qa/data/page_images}"
DOC_CACHE="$PAGE_CACHE/$DOC_ID"

if [ ! -d "$DOC_CACHE" ] || [ -z "$(ls -1 "$DOC_CACHE"/page-*.png 2>/dev/null)" ]; then
    echo "No cached images for $DOC_ID; run run_checkbox_qa.py to populate cache." >&2
    exit 1
fi

FIRST_PAGE=$(ls -1v "$DOC_CACHE"/page-*.png | head -n 1)
if [ -z "$FIRST_PAGE" ]; then
    echo "Failed to locate first page image for $DOC_ID" >&2
    exit 1
fi

IMG_DIR=$(dirname "$FIRST_PAGE")
IMG_NAME=$(basename "$FIRST_PAGE")

GRANITE_DIR="$REPO_ROOT/examples/perception/doc_ai/granite_docling"

docker run --rm --gpus all \
    -v "$GRANITE_DIR:/workspace" \
    -v "$IMG_DIR:/inputs:ro" \
    -v "$OUTPUT_DIR:/outputs" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -w /workspace \
    granite_docling \
    python3 predict.py "/inputs/$IMG_NAME" --output "/outputs/$OUTPUT_NAME"
