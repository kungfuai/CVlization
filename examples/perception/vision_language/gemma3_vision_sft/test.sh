#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running Gemma-3 Vision SFT smoke test..."
echo "This will train for just 30 steps on 100 samples to verify the setup works."
echo ""

# Run training with default config (which is already set for quick testing)
bash "${SCRIPT_DIR}/train.sh"

# Check if model was saved
if [ -d "${SCRIPT_DIR}/outputs/final_model" ]; then
    echo ""
    echo "✓ Smoke test passed!"
    echo "  - Training completed successfully"
    echo "  - Model saved to: ${SCRIPT_DIR}/outputs/final_model"
    echo ""
    echo "To run full training, edit config.yaml:"
    echo "  - Remove or increase max_samples"
    echo "  - Increase max_steps or set num_train_epochs: 2"
    echo "  - For Gemma-3N, change model.name to 'unsloth/gemma-3n-E4B'"
    echo "    and set lora.r: 32, model.chat_template: 'gemma-3n'"
else
    echo ""
    echo "✗ Smoke test failed: Model not found"
    exit 1
fi
