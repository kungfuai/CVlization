#!/bin/bash
# Download HunyuanVideo-1.5 model weights to HuggingFace cache
#
# Downloads to ~/.cache/huggingface/hub/ (standard HF cache location)
# Models are shared across projects and not duplicated.
#
# Required models (~50GB total):
# - tencent/HunyuanVideo-1.5 (480p T2V variant only, ~32GB)
# - Qwen/Qwen2.5-VL-7B-Instruct (~15GB)
# - google/byt5-small (~1GB)
#
# Optional (for I2V mode, requires HF token):
# - black-forest-labs/FLUX.1-Redux-dev

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Load HF_TOKEN from .env if available
if [ -f "${REPO_ROOT}/.env" ]; then
    source "${REPO_ROOT}/.env"
fi

echo "Downloading HunyuanVideo-1.5 model weights to HuggingFace cache"
echo "Location: ~/.cache/huggingface/hub/"
echo "Total: ~50GB"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub CLI..."
    pip install -U "huggingface_hub[cli]"
fi

echo "=== Step 1/4: Downloading HunyuanVideo-1.5 (480p T2V variant) ==="
# Downloads to cache, returns cache path
huggingface-cli download tencent/HunyuanVideo-1.5 \
    --include "config.json" "transformer/480p_t2v_distilled/*" "vae/*" "scheduler/*"

echo ""
echo "=== Step 2/4: Downloading text encoder (Qwen2.5-VL-7B-Instruct) ==="
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct

echo ""
echo "=== Step 3/4: Downloading byT5 encoder ==="
huggingface-cli download google/byt5-small

# Glyph-SDXL-v2 - only on ModelScope, download inside Docker (has Python 3.11)
echo ""
echo "Note: Glyph-SDXL-v2 will be downloaded inside Docker container on first run."

echo ""
echo "=== Step 4/4: Vision encoder (optional, for I2V mode) ==="
if [ -n "${HF_TOKEN}" ]; then
    echo "HF_TOKEN detected. Downloading vision encoder..."
    huggingface-cli download black-forest-labs/FLUX.1-Redux-dev --token "${HF_TOKEN}"
else
    echo "HF_TOKEN not set. Skipping vision encoder (only needed for I2V mode)."
    echo "To download: huggingface-cli download black-forest-labs/FLUX.1-Redux-dev --token <token>"
fi

echo ""
echo "=== Download complete! ==="
echo ""
echo "Cache location:"
du -sh ~/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5 2>/dev/null || echo "  tencent/HunyuanVideo-1.5: (downloading...)"
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct 2>/dev/null || echo "  Qwen/Qwen2.5-VL-7B-Instruct: (downloading...)"
du -sh ~/.cache/huggingface/hub/models--google--byt5-small 2>/dev/null || echo "  google/byt5-small: (downloading...)"
