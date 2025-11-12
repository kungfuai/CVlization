#!/bin/bash
set -e

# CheckboxQA Benchmark Runner
# Usage: ./run_benchmark.sh [model_names...] [--max-docs N]
# Example: ./run_benchmark.sh moondream2
# Example: ./run_benchmark.sh moondream2 florence_2_base --max-docs 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check dependencies
if ! python3 -c "import tqdm" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q tqdm
fi

# Run Python benchmark script
python3 run_checkpoint_qa.py "$@"
