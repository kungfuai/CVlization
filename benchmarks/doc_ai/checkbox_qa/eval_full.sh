#!/bin/bash
# Full evaluation on all 88 documents (579 questions)
# WARNING: This takes approximately 10 hours per model!
#
# Usage:
#   ./eval_full.sh moondream2
#   ./eval_full.sh moondream2 florence_2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name> [model_name...]"
    echo ""
    echo "WARNING: Full evaluation takes ~10 hours per model!"
    echo "Consider using ./eval_dev.sh or ./eval_test.sh for faster runs."
    echo ""
    echo "Available models:"
    ls -1 adapters/*.sh 2>/dev/null | xargs -I{} basename {} .sh | sed 's/^/  /'
    exit 1
fi

echo "Running FULL evaluation (88 docs, 579 questions)"
echo "This will take approximately 10 hours per model."
echo ""

./evaluate.sh "$@"
