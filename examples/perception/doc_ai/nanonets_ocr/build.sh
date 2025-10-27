#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build from the script's directory, works from anywhere
echo "Building Nanonets-OCR2-3B container..."
docker build -t nanonets_ocr "$SCRIPT_DIR"

echo "Build complete! Image: nanonets_ocr"
echo ""
echo "Usage:"
echo "  ./predict.sh [options]"
echo ""
echo "Examples:"
echo "  ./predict.sh --input document.png"
echo "  ./predict.sh --input form.jpg --format json --output result.json"
echo "  ./predict.sh --input chart.png --mode vqa --question 'What is the trend?'"
