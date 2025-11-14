#!/bin/bash
# Multi-page adapter for Qwen3-VL-2B with full PDF support

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

REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Load centralized prompt templates
source "$SCRIPT_DIR/../prompts.sh"
ENHANCED_PROMPT="${ACTIVE_PROMPT}${QUESTION}"

# Convert PDF to absolute path
PDF_ABS=$(realpath "$PDF_PATH")

# Get absolute paths for output
OUTPUT_ABS=$(realpath -m "$OUTPUT_FILE")
OUTPUT_DIR=$(dirname "$OUTPUT_ABS")
OUTPUT_NAME=$(basename "$OUTPUT_ABS")

mkdir -p "$OUTPUT_DIR"

# Get document name and page cache directory
DOC_NAME=$(basename "$PDF_PATH" .pdf)
PAGE_CACHE_ROOT="${CHECKBOX_QA_PAGE_CACHE:-$REPO_ROOT/benchmarks/doc_ai/checkbox_qa/data/page_images}"
DOC_CACHE="$PAGE_CACHE_ROOT/$DOC_NAME"
mkdir -p "$DOC_CACHE"

PAGE_FILES=($(ls -1v "$DOC_CACHE"/page-*.png 2>/dev/null))
if [ ${#PAGE_FILES[@]} -eq 0 ]; then
    echo "No cached page images for $DOC_NAME; run run_checkbox_qa.py to generate cache." >&2
    exit 1
fi
echo "Found ${#PAGE_FILES[@]} cached pages for $DOC_NAME"

if command -v identify >/dev/null 2>&1; then
    identify "${PAGE_FILES[0]}" 2>/dev/null || true
fi

if [ -n "${MAX_PAGES:-}" ] && [ "${#PAGE_FILES[@]}" -gt "$MAX_PAGES" ]; then
    echo "Limiting to first $MAX_PAGES pages for $DOC_NAME"
    PAGE_FILES=("${PAGE_FILES[@]:0:$MAX_PAGES}")
fi

echo "Converted ${#PAGE_FILES[@]} pages to images"

# Build --images argument with all page paths (using container paths)
IMAGES_ARG=""
for page_file in "${PAGE_FILES[@]}"; do
    # Use /inputs/ path as it will be inside the container
    IMAGES_ARG="$IMAGES_ARG /inputs/$(basename "$page_file")"
done

echo "Passing ${#PAGE_FILES[@]} images to VLM"

CLIENT_SCRIPT="$SCRIPT_DIR/../openai_vlm_request.py"
if [ -n "${QWEN3_VL_API_BASE:-}" ]; then
    MODEL_NAME="${QWEN3_VL_SERVE_MODEL:-qwen3-vl-2b}"
    python3 "$CLIENT_SCRIPT" \
        --api-base "$QWEN3_VL_API_BASE" \
        --model "$MODEL_NAME" \
        --prompt "$ENHANCED_PROMPT" \
        --images "${PAGE_FILES[@]}" \
        --output "$OUTPUT_ABS"
    exit 0
fi

# Run Qwen3-VL-2B with multi-image support
QWEN_DIR="$REPO_ROOT/examples/perception/vision_language/qwen3_vl"
DOCKER_ENVS=()
if [ -n "${QWEN3_VL_DEVICE:-}" ]; then
    DOCKER_ENVS+=(-e "QWEN3_VL_DEVICE=$QWEN3_VL_DEVICE")
fi

docker run --runtime nvidia --rm \
    -v "$QWEN_DIR:/workspace" \
    -v "$DOC_CACHE:/inputs:ro" \
    -v "$OUTPUT_DIR:/outputs" \
    -v "$REPO_ROOT:/cvlization_repo:ro" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e PYTHONPATH=/cvlization_repo \
    -e HF_TOKEN=$HF_TOKEN \
    "${DOCKER_ENVS[@]}" \
    -e PROMPT="$ENHANCED_PROMPT" \
    -e QWEN3_VL_VARIANT=2b \
    qwen3-vl \
    bash -c "python3 predict.py --images $IMAGES_ARG --output /outputs/$OUTPUT_NAME --task vqa --prompt \"\$PROMPT\""

echo "Multi-page processing complete"
