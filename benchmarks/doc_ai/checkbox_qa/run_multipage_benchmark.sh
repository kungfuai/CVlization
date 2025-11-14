#!/bin/bash
# Run CheckboxQA benchmark with multi-page PDF support
# Uses all pages of each PDF as input to the VLM

set -e

cd "$(dirname "$0")"

SUBSET_FILE="${1:-data/subset_dev.jsonl}"

echo "================================="
echo "CheckboxQA Multi-Page Benchmark"
echo "================================="
echo "Subset: $SUBSET_FILE"
echo

# First, add multipage adapters to config if not already present
if ! grep -q "qwen3_vl_2b_multipage" config.yaml; then
    echo "  qwen3_vl_2b_multipage:" >> config.yaml
    echo "    adapter: \"adapters/qwen3_vl_2b_multipage.sh\"" >> config.yaml
    echo "    description: \"Qwen3-VL-2B with multi-page PDF support\"" >> config.yaml
    echo "" >> config.yaml
fi

if ! grep -q "qwen3_vl_4b_multipage" config.yaml; then
    echo "  qwen3_vl_4b_multipage:" >> config.yaml
    echo "    adapter: \"adapters/qwen3_vl_4b_multipage.sh\"" >> config.yaml
    echo "    description: \"Qwen3-VL-4B with multi-page PDF support\"" >> config.yaml
    echo "" >> config.yaml
fi

echo "Running qwen3_vl_2b_multipage and qwen3_vl_4b_multipage..."
./run_checkbox_qa.py \
    qwen3_vl_2b_multipage \
    qwen3_vl_4b_multipage \
    --subset "$SUBSET_FILE" \
    --gold data/subset_dev.jsonl

echo
echo "================================="
echo "Benchmark Complete!"
echo "================================="
echo
echo "Results saved in results/ directory"
