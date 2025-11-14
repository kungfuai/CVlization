#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Single-stage build optimized for A10 GPU (compute capability 8.6)
# Final image: ~20GB
#
# For smaller image (~10GB), use: bash build_multistage.sh
#
# Optional build arguments:
#   --build-arg TORCH_CUDA_ARCH_LIST="8.6"  # CUDA architecture (default: 8.6 for A10)
#   --build-arg MAX_JOBS=8                  # Parallel compilation jobs (default: 4)
#
# To see verbose compilation logs:
#   VERBOSE=1 bash build.sh
#
# To use more CPU cores for faster compilation:
#   MAX_JOBS=8 bash build.sh

# Set progress mode (plain for verbose, auto for quiet)
PROGRESS="${VERBOSE:+plain}"
PROGRESS="${PROGRESS:-auto}"

# Allow overriding MAX_JOBS via environment variable
MAX_JOBS_ARG="${MAX_JOBS:+--build-arg MAX_JOBS=$MAX_JOBS}"

docker build \
    --progress="$PROGRESS" \
    $MAX_JOBS_ARG \
    -t egstalker \
    "$SCRIPT_DIR"
