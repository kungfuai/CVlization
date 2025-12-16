#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"

mkdir -p "${MODELS_DIR}"

echo "=========================================="
echo "RealVideo Model Download"
echo "=========================================="
echo ""
echo "This will download the following models:"
echo "  1. Wan2.2-S2V-14B (~50GB) - Base video model"
echo "  2. zai-org/RealVideo model.pt (~2GB) - Lip-sync checkpoint"
echo ""
echo "Total: ~52GB disk space required"
echo ""

# Download Wan2.2-S2V-14B
echo "[1/2] Downloading Wan2.2-S2V-14B..."
if [ ! -d "${MODELS_DIR}/Wan2.2-S2V-14B" ]; then
    huggingface-cli download Wan-AI/Wan2.2-S2V-14B \
        --local-dir-use-symlinks False \
        --local-dir "${MODELS_DIR}/Wan2.2-S2V-14B"
    echo "Wan2.2-S2V-14B downloaded."
else
    echo "Wan2.2-S2V-14B already exists, skipping."
fi

# Download RealVideo checkpoint
echo "[2/2] Downloading RealVideo checkpoint..."
if [ ! -f "${MODELS_DIR}/model.pt" ]; then
    huggingface-cli download zai-org/RealVideo model.pt \
        --local-dir-use-symlinks False \
        --local-dir "${MODELS_DIR}"
    echo "RealVideo checkpoint downloaded."
else
    echo "RealVideo checkpoint already exists, skipping."
fi

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Models saved to: ${MODELS_DIR}"
echo ""
echo "Update config/config.py to point to:"
echo "  PATH_TO_YOUR_MODEL = '/workspace/models/model.pt'"
echo ""
