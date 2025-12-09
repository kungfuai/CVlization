#!/bin/bash
# Evaluation on test subset (20 documents, ~138 questions)
#
# Usage:
#   ./eval_test.sh moondream2
#   ./eval_test.sh moondream2 florence_2

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

./evaluate.sh "$@" --subset test
