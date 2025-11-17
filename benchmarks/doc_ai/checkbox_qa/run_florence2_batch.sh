#!/usr/bin/env bash
#
# Wrapper script for run_florence2_batch.py
# Run Florence-2 on CheckboxQA benchmark
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 run_florence2_batch.py "$@"
