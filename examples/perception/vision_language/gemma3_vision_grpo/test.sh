#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running Gemma-3 Vision GRPO smoke test..."
echo "This will train for 60 steps on 50 samples to verify GRPO setup works."
echo ""

# Run training with default config (which is already set for quick testing)
bash "${SCRIPT_DIR}/train.sh"

# Check if model was saved
if [ -d "${SCRIPT_DIR}/outputs/final_model" ]; then
    echo ""
    echo "✓ Smoke test passed!"
    echo "  - GRPO training completed successfully"
    echo "  - Model saved to: ${SCRIPT_DIR}/outputs/final_model"
    echo ""
    echo "To run full training, edit config.yaml:"
    echo "  - Remove max_samples"
    echo "  - Increase max_steps or set num_train_epochs: 2"
    echo "  - Increase num_generations if you have more VRAM"
    echo "  - Adjust reward values for your task"
else
    echo ""
    echo "✗ Smoke test failed: Model not found"
    exit 1
fi
