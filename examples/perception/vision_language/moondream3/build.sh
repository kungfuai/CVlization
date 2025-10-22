#!/bin/bash
set -e

echo "Building Moondream3 Docker image..."
docker build -t moondream3 examples/doc_ai/moondream3/

echo "Build complete!"
echo ""
echo "To run inference:"
echo "  bash examples/doc_ai/moondream3/predict.sh --image path/to/image.jpg"
