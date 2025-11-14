#!/bin/bash
# Convenience wrapper to download CheckboxQA PDFs inside the benchmark Docker image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${CHECKBOX_QA_IMAGE:-checkbox_qa}"
JSON_PATH="${1:-data/document_url_map.json}"
OUT_DIR="${2:-data/documents}"

echo "Using Docker image: $IMAGE"
echo "JSON map: $JSON_PATH"
echo "Output dir: $OUT_DIR"

docker run --rm \
    -v "$SCRIPT_DIR":/workspace \
    -w /workspace \
    "$IMAGE" \
    python download_documents.py --json_path "$JSON_PATH" --out_dir "$OUT_DIR"
