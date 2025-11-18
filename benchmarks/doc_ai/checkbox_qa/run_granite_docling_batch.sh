#!/usr/bin/env bash
# Convenience wrapper for Granite-Docling batch inference on CheckboxQA
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 run_granite_docling_batch.py "$@"
