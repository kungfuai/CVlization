#!/bin/bash
# Multi-page adapter for Phi-4-Multimodal ensuring all pages are passed at once.

set -euo pipefail

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
source "$SCRIPT_DIR/../prompts.sh"
ENHANCED_PROMPT="${ACTIVE_PROMPT}${QUESTION}"

PDF_ABS=$(realpath "$PDF_PATH")
OUTPUT_ABS=$(realpath -m "$OUTPUT_FILE")
OUTPUT_DIR=$(dirname "$OUTPUT_ABS")
OUTPUT_NAME=$(basename "$OUTPUT_ABS")
mkdir -p "$OUTPUT_DIR"

DOC_NAME=$(basename "$PDF_PATH" .pdf)
PAGE_CACHE_ROOT="${CHECKBOX_QA_PAGE_CACHE:-$REPO_ROOT/benchmarks/doc_ai/checkbox_qa/data/page_images}"
DOC_CACHE="$PAGE_CACHE_ROOT/$DOC_NAME"
mkdir -p "$DOC_CACHE"

PAGE_FILES=($(ls -1v "$DOC_CACHE"/page-*.png 2>/dev/null))
if [ ${#PAGE_FILES[@]} -eq 0 ]; then
    echo "No cached page images for $DOC_NAME; run run_checkbox_qa.py to generate cache." >&2
    exit 1
fi

if command -v identify >/dev/null 2>&1; then
    identify "${PAGE_FILES[0]}" 2>/dev/null || true
fi

if [ -n "${PHI4_MAX_PAGES:-}" ] && [ "${#PAGE_FILES[@]}" -gt "$PHI4_MAX_PAGES" ]; then
    echo "Limiting to first $PHI4_MAX_PAGES pages for $DOC_NAME"
    PAGE_FILES=("${PAGE_FILES[@]:0:$PHI4_MAX_PAGES}")
fi

TMP_DIR="/tmp/checkbox_qa_phi_${DOC_NAME}_$$"
mkdir -p "$TMP_DIR"
COMBINED_IMG="$TMP_DIR/${DOC_NAME}_combined.png"

python3 - "${PAGE_FILES[@]}" "$COMBINED_IMG" <<'PY'
import sys
from pathlib import Path
from PIL import Image

*page_paths, combined = sys.argv[1:]
images = [Image.open(p).convert("RGB") for p in page_paths]
if not images:
    raise SystemExit("No images supplied for combination")

width = max(im.width for im in images)
total_height = sum(im.height for im in images)
canvas = Image.new("RGB", (width, total_height), (255, 255, 255))

y = 0
for img in images:
    if img.width != width:
        ratio = width / img.width
        img = img.resize((width, int(img.height * ratio)), Image.BICUBIC)
    canvas.paste(img, (0, y))
    y += img.height

Path(combined).parent.mkdir(parents=True, exist_ok=True)
canvas.save(combined)
PY

PHI_DIR="$REPO_ROOT/examples/perception/vision_language/phi_4_multimodal_instruct"

CLIENT_SCRIPT="$SCRIPT_DIR/../openai_vlm_request.py"
if [ -n "${PHI4_API_BASE:-}" ]; then
    MODEL_NAME="${PHI4_SERVE_MODEL:-phi-4-multimodal}"
    python3 "$CLIENT_SCRIPT" \
        --api-base "$PHI4_API_BASE" \
        --model "$MODEL_NAME" \
        --prompt "$ENHANCED_PROMPT" \
        --images "${PAGE_FILES[@]}" \
        --output "$OUTPUT_ABS"
    rm -rf "$TMP_DIR"
    echo "Phi-4 multipage complete for $DOC_NAME"
    exit 0
fi

docker run --runtime nvidia --rm \
    -v "$PHI_DIR:/workspace" \
    -v "$TMP_DIR:/inputs:ro" \
    -v "$OUTPUT_DIR:/outputs" \
    -v "$REPO_ROOT:/cvlization_repo:ro" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e PYTHONPATH=/cvlization_repo \
    -e HF_TOKEN=$HF_TOKEN \
    -e PROMPT="$ENHANCED_PROMPT" \
    phi-4-multimodal-instruct \
    python3 predict.py \
        --image "/inputs/$(basename "$COMBINED_IMG")" \
        --output "/outputs/$OUTPUT_NAME" \
        --prompt "$ENHANCED_PROMPT"

rm -rf "$TMP_DIR"
echo "Phi-4 multipage complete for $DOC_NAME"
