#!/bin/bash
# Download models and checkpoints for EGSTalker
# This script downloads necessary models if they're available publicly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"

mkdir -p "${MODELS_DIR}"

echo "EGSTalker Model Download Script"
echo "=============================="
echo ""
echo "Note: Pre-trained models may not be publicly available."
echo "Please check the EGSTalker repository for model download links."
echo ""
echo "If you have trained your own model, place it in:"
echo "  ${MODELS_DIR}/"
echo ""
echo "Model directory structure should be:"
echo "  models/"
echo "    iteration_10000/"
echo "      point_cloud/"
echo "      ..."
echo ""

# Check if models directory has checkpoints
if [ -d "${MODELS_DIR}" ] && [ "$(ls -A ${MODELS_DIR} 2>/dev/null)" ]; then
    echo "Found existing models in ${MODELS_DIR}"
    ls -lh "${MODELS_DIR}"
else
    echo "No models found. Please:"
    echo "1. Train your own model using the EGSTalker training script"
    echo "2. Or download pre-trained models from the official repository"
    echo "3. Place them in: ${MODELS_DIR}/"
fi

echo ""
echo "For training instructions, see:"
echo "  https://github.com/ZhuTianheng/EGSTalker"
echo ""
