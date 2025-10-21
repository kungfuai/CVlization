#!/bin/bash

# Default values
IMAGE_PATH="examples/sample.jpg"
OUTPUT_PATH="outputs/result.txt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

docker run --runtime nvidia \
    -v $(pwd)/examples/doc_ai/dots-ocr:/workspace \
    -v $(pwd)/data/container_cache:/root/.cache \
    -e HF_TOKEN=$HF_TOKEN \
    dots_ocr \
    python3 predict.py --image "$IMAGE_PATH" --output "$OUTPUT_PATH" "$@"
