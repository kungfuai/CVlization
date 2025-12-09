#!/bin/bash
# Quick evaluation on dev subset (5 documents, ~40 questions)
# Approximately 4 minutes per model
#
# Usage:
#   ./eval_dev.sh moondream2
#   ./eval_dev.sh moondream2 florence_2 qwen3_vl_2b

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name> [model_name...]"
    echo ""
    echo "Available models:"
    ls -1 adapters/*.sh 2>/dev/null | xargs -I{} basename {} .sh | sed 's/^/  /'
    exit 1
fi

./evaluate.sh "$@" --subset dev
