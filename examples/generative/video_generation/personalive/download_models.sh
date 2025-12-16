#!/usr/bin/env bash
set -euo pipefail

# Pre-populate HuggingFace cache with PersonaLive model weights.
# This is OPTIONAL - models are downloaded lazily on first run.
# Cache location: ~/.cache/huggingface (shared across all examples)

echo "=== PersonaLive Model Pre-Download ==="
echo "Pre-populating HuggingFace cache..."
echo ""

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-cli..."
    pip install -q huggingface_hub[cli]
fi

# Download PersonaLive weights (~2GB)
echo "=== Downloading PersonaLive weights ==="
huggingface-cli download huaichang/PersonaLive

# Download SD Image Variations base model (~5GB)
echo ""
echo "=== Downloading SD Image Variations (base model) ==="
huggingface-cli download lambdalabs/sd-image-variations-diffusers

# Download SD VAE (~350MB)
echo ""
echo "=== Downloading SD VAE ==="
huggingface-cli download stabilityai/sd-vae-ft-mse

echo ""
echo "=== Download Complete ==="
echo ""
echo "Models cached in: ${HF_HOME:-~/.cache/huggingface}"
echo ""
echo "You can now run: ./predict.sh"
