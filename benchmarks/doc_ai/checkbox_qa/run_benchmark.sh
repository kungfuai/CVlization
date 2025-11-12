#!/bin/bash
set -e

# CheckboxQA Benchmark Runner
# Usage: ./run_benchmark.sh [model_names...]
# Example: ./run_benchmark.sh florence_2_base qwen3_vl_2b

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "CheckboxQA Benchmark"
echo "=========================================="
echo ""

# Check dependencies
if ! python3 -c "import datasets" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
fi

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Results directory: $RESULTS_DIR"
echo ""

# Load dataset and get document list
echo "Loading CheckboxQA dataset..."
DOCS=$(python3 dataset_builder.py 2>/dev/null | grep "First document:" | awk '{print $3}')

if [ -z "$DOCS" ]; then
    echo "Error: Failed to load dataset"
    exit 1
fi

echo "Dataset loaded successfully"
echo ""

# TODO: Implement full benchmark runner
# This is a placeholder for the full implementation
# The complete version would:
# 1. Iterate through each model
# 2. For each document, extract questions
# 3. Call model adapter with PDF + question
# 4. Collect all predictions into predictions.jsonl
# 5. Run evaluate.py to compute ANLS* score
# 6. Generate leaderboard.md

echo "NOTE: Full benchmark runner not yet implemented"
echo "To evaluate a model manually:"
echo "  1. Generate predictions in results/\${TIMESTAMP}/\${MODEL}/predictions.jsonl"
echo "  2. Run: python3 evaluate.py --pred results/\${TIMESTAMP}/\${MODEL}/predictions.jsonl --gold data/gold.jsonl"
echo ""
echo "Example prediction format (see data/GPT.jsonl for reference)"
