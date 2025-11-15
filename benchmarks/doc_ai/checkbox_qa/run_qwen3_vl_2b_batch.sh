#!/usr/bin/env bash
# Convenience wrapper for run_qwen3_vl_2b_batch.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 run_qwen3_vl_2b_batch.py "$@"
