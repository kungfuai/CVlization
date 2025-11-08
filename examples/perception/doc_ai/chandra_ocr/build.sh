#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building Chandra OCR Docker image..."
docker build -t chandra-ocr "$SCRIPT_DIR"

echo "Build complete! Image: chandra-ocr"
