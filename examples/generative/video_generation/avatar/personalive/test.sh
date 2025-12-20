#!/usr/bin/env bash
set -euo pipefail

# Smoke test for PersonaLive
# Uses demo assets with reduced frame count for quick validation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== PersonaLive Smoke Test ==="
echo ""

# Run with minimal settings for quick test
# Note: steps must be divisible by temporal_adaptive_step (4)
"${SCRIPT_DIR}/predict.sh" \
    --ref_image demo/ref_image.png \
    --driving_video demo/driving_video.mp4 \
    --max_frames 8 \
    --steps 4 \
    --output outputs/test_output.mp4 \
    "$@"

# Check output
if [[ -f "${SCRIPT_DIR}/outputs/test_output.mp4" ]]; then
    echo ""
    echo "=== Test Passed ==="
    echo "Output: ${SCRIPT_DIR}/outputs/test_output.mp4"
    ls -lh "${SCRIPT_DIR}/outputs/test_output.mp4"
else
    echo ""
    echo "=== Test Failed ==="
    echo "Output file not found"
    exit 1
fi
