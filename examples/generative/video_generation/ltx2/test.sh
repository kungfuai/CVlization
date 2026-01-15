#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running LTX-2 smoke test..."
echo "Using distilled pipeline with minimal frames for fast testing"

# Run with minimal settings for quick test
"${SCRIPT_DIR}/predict.sh" \
    --pipeline distilled \
    --num-frames 33 \
    --height 512 \
    --width 768 \
    --prompt "A cat sitting on a windowsill watching birds outside" \
    --output test_output.mp4

echo ""
echo "Test complete! Check test_output.mp4"
