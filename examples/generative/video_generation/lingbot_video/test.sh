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
    --prompt "A young woman with long brown hair standing in a bright modern apartment living room. She is wearing an oversized cream-colored knit cardigan over a white tank top paired with high-waisted beige trousers. The background features a beige sofa and large windows with soft natural light. The camera is stationary at eye level with a medium shot." \
    --output test_output.mp4

echo ""
echo "Test complete! Check test_output.mp4"
