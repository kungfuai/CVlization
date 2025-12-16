#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running TurboDiffusion smoke test..."
echo "Note: First run will download ~8GB of model weights."
echo ""

bash "${SCRIPT_DIR}/predict.sh" \
    --prompt "A golden retriever running on a beach at sunset, slow motion" \
    --output outputs/test_output.mp4 \
    --num_steps 4 \
    --resolution 480p \
    --seed 42

echo ""
echo "Smoke test complete!"
echo "Output: ${SCRIPT_DIR}/outputs/test_output.mp4"
