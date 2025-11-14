#!/bin/bash
# Evaluate predictions JSONL via the benchmark Docker image.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <predictions.jsonl> [gold.jsonl]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${CHECKBOX_QA_IMAGE:-checkbox_qa}"
PRED="$1"
GOLD="${2:-data/gold.jsonl}"

docker run --rm \
    -v "$SCRIPT_DIR":/workspace \
    -w /workspace \
    "$IMAGE" \
    python evaluate.py --pred "$PRED" --gold "$GOLD"
