#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root and set cache directory
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
CACHE_DIR="$REPO_ROOT/data/container_cache"

# Default values (relative to script directory)
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

# Create outputs directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/outputs"

# Run from script directory, works from anywhere
# CVL dual-mode: If CVL_INPUTS/CVL_OUTPUTS are set (by user or cvl run), mount and use them
# Otherwise, use workspace-relative paths (existing behavior)
DOCKER_MOUNTS="-v $SCRIPT_DIR:/workspace -v $CACHE_DIR:/root/.cache"
DOCKER_ENVS="-e HF_TOKEN=$HF_TOKEN"

if [ -n "$CVL_INPUTS" ]; then
    # User or cvl set CVL_INPUTS - mount it
    DOCKER_MOUNTS="$DOCKER_MOUNTS -v $CVL_INPUTS:/mnt/cvl/inputs:ro"
    DOCKER_ENVS="$DOCKER_ENVS -e CVL_INPUTS=/mnt/cvl/inputs"
fi

if [ -n "$CVL_OUTPUTS" ]; then
    # User or cvl set CVL_OUTPUTS - mount it
    mkdir -p "$CVL_OUTPUTS"
    DOCKER_MOUNTS="$DOCKER_MOUNTS -v $CVL_OUTPUTS:/mnt/cvl/outputs"
    DOCKER_ENVS="$DOCKER_ENVS -e CVL_OUTPUTS=/mnt/cvl/outputs"
fi

docker run --runtime nvidia \
    $DOCKER_MOUNTS \
    $DOCKER_ENVS \
    moondream2 \
    python3 predict.py --image "$IMAGE_PATH" --output "$OUTPUT_PATH" "$@"
