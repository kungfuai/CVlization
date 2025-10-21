#!/bin/bash

# Default values
IMAGE_PATH="examples/sample.jpg"
OUTPUT_PATH="outputs/result.txt"

# Optimized batch sizes for A10 GPU (23GB VRAM)
# Can tune these upward if you have more VRAM
export RECOGNITION_BATCH_SIZE=${RECOGNITION_BATCH_SIZE:-32}
export DETECTOR_BATCH_SIZE=${DETECTOR_BATCH_SIZE:-4}
export LAYOUT_BATCH_SIZE=${LAYOUT_BATCH_SIZE:-4}

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
    -v $(pwd)/examples/doc_ai/surya:/workspace \
    -v $(pwd)/data/container_cache:/root/.cache \
    -e RECOGNITION_BATCH_SIZE=$RECOGNITION_BATCH_SIZE \
    -e DETECTOR_BATCH_SIZE=$DETECTOR_BATCH_SIZE \
    -e LAYOUT_BATCH_SIZE=$LAYOUT_BATCH_SIZE \
    surya \
    python3 predict.py --image "$IMAGE_PATH" --output "$OUTPUT_PATH" "$@"
