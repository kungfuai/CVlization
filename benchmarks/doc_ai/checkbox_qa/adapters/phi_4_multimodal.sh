#!/bin/bash
# Adapter for Phi-4-Multimodal on CheckboxQA
# Usage: ./phi_4_multimodal.sh <pdf_path> <question> --output <output_file>

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

# Use cached page images
DOC_ID="$(basename "$PDF_PATH" .pdf)"
CACHE_ROOT="${CHECKBOX_QA_PAGE_CACHE:-$REPO_ROOT/benchmarks/doc_ai/checkbox_qa/data/page_images}"
DOC_CACHE="$CACHE_ROOT/$DOC_ID"

if [ ! -d "$DOC_CACHE" ] || [ -z "$(ls -A "$DOC_CACHE"/page-*.png 2>/dev/null)" ]; then
    echo "No cached images for $DOC_ID; run run_checkbox_qa.py to generate cache." >&2
    exit 1
fi

FIRST_PAGE=$(ls -1v "$DOC_CACHE"/page-*.png | head -n 1)
IMG_DIR=$(dirname "$FIRST_PAGE")
IMG_NAME=$(basename "$FIRST_PAGE")

# Run Phi-4-Multimodal with question as custom prompt
# Follow CVlization pattern: mount repo for cvlization package
PHI4_DIR="$REPO_ROOT/examples/perception/vision_language/phi_4_multimodal_instruct"

docker run --runtime nvidia --rm \
    -v "$PHI4_DIR:/workspace" \
    -v "$IMG_DIR:/inputs:ro" \
    -v "$OUTPUT_DIR:/outputs" \
    -v "$REPO_ROOT:/cvlization_repo:ro" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e PYTHONPATH=/cvlization_repo \
    -e HF_TOKEN=$HF_TOKEN \
    phi-4-multimodal-instruct \
    python3 predict.py \
        --image "/inputs/$IMG_NAME" \
        --output "/outputs/$OUTPUT_NAME" \
        --prompt "$QUESTION"

# Cache is reused; no cleanup
