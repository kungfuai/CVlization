#!/bin/bash
set -e

# Build the Nanonets-OCR Docker image
echo "Building Nanonets-OCR2-3B container..."
docker build -t nanonets-ocr .

echo "Build complete! Image: nanonets-ocr"
echo ""
echo "Usage:"
echo "  ./predict.sh <input_image> [options]"
echo ""
echo "Examples:"
echo "  ./predict.sh document.png"
echo "  ./predict.sh form.jpg --format json --output result.json"
echo "  ./predict.sh chart.png --mode vqa --question 'What is the trend?'"
