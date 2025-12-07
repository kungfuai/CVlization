#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running HunyuanVideo-1.5 smoke test..."
echo "Note: First run will download ~25GB of model weights."
echo ""

# Quick smoke test with minimal frames
# Using 480p T2V with CFG distillation for faster inference
bash "${SCRIPT_DIR}/predict.sh" \
    --prompt "A cat walking slowly in a garden, cinematic lighting" \
    --resolution 480p \
    --video_length 25 \
    --num_inference_steps 20 \
    --cfg_distilled \
    --no_sr \
    --output outputs/test_output.mp4

echo ""
echo "Smoke test complete!"
echo "Output: ${SCRIPT_DIR}/outputs/test_output.mp4"
