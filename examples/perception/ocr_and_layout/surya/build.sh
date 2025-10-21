#!/bin/bash
set -e

echo "Building Surya OCR Docker image..."
docker build -t surya examples/doc_ai/surya/

echo "Build complete!"
echo ""
echo "To run inference:"
echo "  bash examples/doc_ai/surya/predict.sh --image path/to/image.jpg"
