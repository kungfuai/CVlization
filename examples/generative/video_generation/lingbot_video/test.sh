#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running LingBot-Video smoke test..."
echo "Using dense-1.3b model with minimal frames for fast testing"

# Run with minimal settings for quick test
"${SCRIPT_DIR}/predict.sh" \
    --model dense-1.3b \
    --mode t2v \
    --num-frames 21 \
    --height 480 \
    --width 832 \
    --steps 20 \
    --prompt "A robotic arm picks up a red cube from a table" \
    --output test_output.mp4

echo ""
echo "Test complete! Check test_output.mp4"
