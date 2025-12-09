#!/bin/bash
# Run CheckboxQA evaluation on one or more models
#
# Usage:
#   ./evaluate.sh moondream2                    # Evaluate single model (full dataset)
#   ./evaluate.sh moondream2 florence_2         # Evaluate multiple models
#   ./evaluate.sh moondream2 --subset dev       # Quick test (5 docs)
#   ./evaluate.sh moondream2 --subset test      # Test subset (20 docs)
#
# The dataset is automatically downloaded on first run.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check dependencies
if ! python3 -c "import tqdm" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q tqdm requests
fi

# Run benchmark
python3 run_checkbox_qa.py "$@"
