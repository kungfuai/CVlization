#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build from the script's directory, works from anywhere
echo "Building dots.ocr container..."
docker build -t dots_ocr "$SCRIPT_DIR"

# Download required model (idempotent - skips if already exists)
# This is optional - the model will be downloaded on first predict.sh run if not cached
echo ""
echo "Downloading required model..."
python3 "$SCRIPT_DIR/download_model.py" || echo "Model pre-download skipped (will download on first run)"

echo "Build complete! Image: dots_ocr"
echo ""
echo "Usage:"
echo "  ./predict.sh [options]"
echo ""
echo "Example:"
echo "  ./predict.sh --image document.jpg"
echo "  ./predict.sh --image scan.jpg --format json"
