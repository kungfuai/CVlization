#!/bin/bash
set -e

# Check if input file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_file> [options]"
    echo ""
    echo "Options:"
    echo "  --output <file>      Save output to file (optional)"
    echo "  --format <format>    Output format: json (default), markdown, or text"
    echo "  --export-tables      Export tables separately"
    echo "  --export-images      Export images separately"
    echo ""
    echo "Examples:"
    echo "  $0 sample.pdf"
    echo "  $0 sample.pdf --format markdown --output result.md"
    echo "  $0 document.jpg --format json --export-tables"
    exit 1
fi

INPUT_FILE="$1"
shift  # Remove first argument, keep the rest as options

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Get absolute path and filename
INPUT_DIR=$(cd "$(dirname "$INPUT_FILE")" && pwd)
INPUT_NAME=$(basename "$INPUT_FILE")

# Create outputs directory
mkdir -p "$(pwd)/outputs"

echo "Processing: $INPUT_FILE"
echo "Using Docling for document extraction..."
echo ""

# Run Docker container
docker run --rm \
    -v "$INPUT_DIR:/app/inputs:ro" \
    -v "$(pwd)/outputs:/app/outputs" \
    docling-serve \
    python predict.py "/app/inputs/$INPUT_NAME" "$@"

echo ""
echo "Done!"
