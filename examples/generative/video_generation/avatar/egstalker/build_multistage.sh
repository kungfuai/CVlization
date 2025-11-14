#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Multi-stage build for smaller image (~10GB instead of ~20GB)
# Build stage compiles CUDA extensions, runtime stage only includes what's needed
#
# Savings:
#   - Removes build tools (gcc, g++, build-essential)
#   - Uses runtime base instead of devel (7.7GB vs 17.3GB)
#   - Final image: ~10.5GB (vs 20.1GB single-stage)
#
# Build arguments:
#   --build-arg TORCH_CUDA_ARCH_LIST="8.6"  # Already optimized for A10
#   --build-arg MAX_JOBS=8                  # Parallel compilation jobs (default: 4)
#
# To see verbose compilation logs:
#   VERBOSE=1 bash build_multistage.sh
#
# To use more CPU cores for faster compilation:
#   MAX_JOBS=8 bash build_multistage.sh

# Set progress mode (plain for verbose, auto for quiet)
PROGRESS="${VERBOSE:+plain}"
PROGRESS="${PROGRESS:-auto}"

# Allow overriding MAX_JOBS via environment variable
MAX_JOBS_ARG="${MAX_JOBS:+--build-arg MAX_JOBS=$MAX_JOBS}"

docker build \
    --progress="$PROGRESS" \
    $MAX_JOBS_ARG \
    -f "$SCRIPT_DIR/Dockerfile.multistage" \
    -t egstalker:multistage \
    "$SCRIPT_DIR"
