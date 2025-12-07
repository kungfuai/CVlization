#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running Kandinsky 5.0 smoke test..."
echo "Note: First run will download ~25GB of model weights."
echo ""

# Use distilled model for faster test (16 steps vs 50)
bash "${SCRIPT_DIR}/predict.sh" \
    --prompt "A cat playing piano, cinematic lighting" \
    --config distilled \
    --duration 5 \
    --offload \
    --output outputs/test_output.mp4

echo ""
echo "Smoke test complete!"
echo "Output: ${SCRIPT_DIR}/outputs/test_output.mp4"
