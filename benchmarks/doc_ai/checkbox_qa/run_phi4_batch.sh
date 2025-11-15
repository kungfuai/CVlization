#!/usr/bin/env bash
# Convenience wrapper for run_phi4_batch.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 run_phi4_batch.py "$@"
