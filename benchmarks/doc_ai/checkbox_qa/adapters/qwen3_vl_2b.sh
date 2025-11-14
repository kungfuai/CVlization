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

DOC_NAME=$(basename "$PDF_PATH" .pdf)
PAGE_CACHE_ROOT="${CHECKBOX_QA_PAGE_CACHE:-$REPO_ROOT/benchmarks/doc_ai/checkbox_qa/data/page_images}"
DOC_CACHE="$PAGE_CACHE_ROOT/$DOC_NAME"
FIRST_PAGE="$DOC_CACHE/page-001.png"

if [ ! -f "$FIRST_PAGE" ]; then
    echo "No cached first page for $DOC_NAME; run run_checkbox_qa.py to generate cache." >&2
    exit 1
fi

IMG_DIR=$(dirname "$FIRST_PAGE")
IMG_NAME=$(basename "$FIRST_PAGE")

CLIENT_SCRIPT="$SCRIPT_DIR/../openai_vlm_request.py"

if [ -n "${QWEN3_VL_API_BASE:-}" ]; then
    MODEL_NAME="${QWEN3_VL_SERVE_MODEL:-qwen3-vl-2b}"
    python3 "$CLIENT_SCRIPT" \
        --api-base "$QWEN3_VL_API_BASE" \
        --model "$MODEL_NAME" \
        --prompt "$QUESTION" \
        --images "$FIRST_PAGE" \
        --output "$OUTPUT_ABS"
    exit 0
fi

# Run Qwen3-VL with question as custom prompt
QWEN_DIR="$REPO_ROOT/examples/perception/vision_language/qwen3_vl"

docker run --runtime nvidia --rm \
    -v "$QWEN_DIR:/workspace" \
    -v "$IMG_DIR:/inputs:ro" \
    -v "$OUTPUT_DIR:/outputs" \
    -v "$REPO_ROOT:/cvlization_repo:ro" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e PYTHONPATH=/cvlization_repo \
    -e HF_TOKEN=$HF_TOKEN \
    -e QWEN3_VL_VARIANT=2b \
    qwen3-vl \
    python3 predict.py \
        --image "/inputs/$IMG_NAME" \
        --output "/outputs/$OUTPUT_NAME" \
        --task vqa \
        --prompt "$QUESTION"

# No cleanup needed; cache persists for future runs
