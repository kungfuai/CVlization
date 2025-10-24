#!/bin/bash
set -e

# Build the Docling Docker image
echo "Building Docling container..."
docker build -t docling_serve .

echo "Build complete! Image: docling_serve"
echo ""
echo "Usage:"
echo "  ./predict.sh <input_file> [options]"
echo ""
echo "Example:"
echo "  ./predict.sh sample.pdf"
echo "  ./predict.sh sample.pdf --format markdown"
