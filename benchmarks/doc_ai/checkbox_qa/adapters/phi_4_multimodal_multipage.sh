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

MAX_PAGES="${MAX_PAGES:-20}"
if [ -n "${MAX_PAGES:-}" ] && [ "${#PAGE_FILES[@]}" -gt "$MAX_PAGES" ]; then
    echo "Limiting to first $MAX_PAGES pages for $DOC_NAME"
    PAGE_FILES=("${PAGE_FILES[@]:0:$MAX_PAGES}")
fi

API_MAX_PAGES="${PHI4_API_MAX_PAGES:-8}"
if [ -n "${PHI4_API_BASE:-}" ] && [ "${#PAGE_FILES[@]}" -gt "$API_MAX_PAGES" ]; then
    echo "API mode: limiting to first $API_MAX_PAGES pages to stay within context window"
    PAGE_FILES=("${PAGE_FILES[@]:0:$API_MAX_PAGES}")
fi

TMP_DIR="/tmp/checkbox_qa_phi_${DOC_NAME}_$$"
mkdir -p "$TMP_DIR"
COMBINED_IMG="$TMP_DIR/${DOC_NAME}_combined.png"
RESIZED_DIR="$TMP_DIR/resized"
mkdir -p "$RESIZED_DIR"
MAX_IMAGE_WIDTH="${PHI4_MAX_IMAGE_WIDTH:-1400}"

if command -v python3 >/dev/null 2>&1; then
    :
else
    echo "python3 is required for resizing images" >&2
    exit 1
fi

mapfile -t RESIZED_FILES < <(
python3 - "$MAX_IMAGE_WIDTH" "$RESIZED_DIR" "${PAGE_FILES[@]}" <<'PY'
import sys
from pathlib import Path
from PIL import Image

max_width = int(sys.argv[1])
out_dir = Path(sys.argv[2])
paths = [Path(p) for p in sys.argv[3:]]

for idx, path in enumerate(paths):
    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = max(1, int(img.height * ratio))
        img = img.resize((max_width, new_height), Image.BICUBIC)
    out_path = out_dir / f"{idx:03d}_{path.name}"
    img.save(out_path, format="PNG")
    print(out_path)
PY
)

if [ "${#RESIZED_FILES[@]}" -eq 0 ]; then
    echo "Failed to prepare resized images for $DOC_NAME" >&2
    exit 1
fi

python3 - "${RESIZED_FILES[@]}" "$COMBINED_IMG" <<'PY'
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
        --images "${RESIZED_FILES[@]}" \
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
phi-4-multimodal-instruct \
python3 predict.py \
        --image "/inputs/$(basename "$COMBINED_IMG")" \
        --output "/outputs/$OUTPUT_NAME" \
        --prompt "$ENHANCED_PROMPT"

rm -rf "$TMP_DIR"
echo "Phi-4 multipage complete for $DOC_NAME"
