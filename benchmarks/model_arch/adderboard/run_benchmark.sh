#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$#" -eq 0 ]; then
  python run_benchmark.py submissions/reference_python_add.py
else
  python run_benchmark.py "$@"
fi
