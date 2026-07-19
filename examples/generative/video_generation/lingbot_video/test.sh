#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running LingBot-Video smoke test..."
echo "Using dense-1.3b model with minimal frames for fast testing"

# Download canonical structured prompt if not present
PROMPT_JSON="canonical_t2v_prompt.json"
if [ ! -f "$PROMPT_JSON" ]; then
    echo "Downloading canonical structured prompt..."
    curl -sL -o "$PROMPT_JSON" \
        "https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_video/canonical_t2v_prompt.json"
fi

# Run with structured prompt (official format, best quality)
"${SCRIPT_DIR}/predict.sh" \
    --model dense-1.3b \
    --mode t2v \
    --num-frames 21 \
    --height 480 \
    --width 832 \
    --steps 20 \
    --prompt-json "$PROMPT_JSON" \
    --output test_output.mp4

echo ""
echo "Test complete! Check test_output.mp4"
