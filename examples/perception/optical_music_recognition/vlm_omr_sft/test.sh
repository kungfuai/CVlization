#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running Gemma-3 OMR SFT smoke test..."
echo "Trains for 30 steps on 100 lieder samples to verify the setup."
echo ""

bash "${SCRIPT_DIR}/train.sh"

if [ -d "${SCRIPT_DIR}/outputs/final_model" ]; then
    echo ""
    echo "Smoke test passed!"
    echo "  Model saved to: ${SCRIPT_DIR}/outputs/final_model"
    echo ""
    echo "For a full training run, edit config.yaml:"
    echo "  - Remove max_samples"
    echo "  - Add quartets / orchestra to corpora"
    echo "  - Replace max_steps with num_train_epochs: 2"
else
    echo ""
    echo "Smoke test failed: outputs/final_model not found"
    exit 1
fi
