#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Single-stage build optimized for A10 GPU (compute capability 8.6)
# Final image: ~20GB
#
# For smaller image (~10GB), use: bash build_multistage.sh
#
# Images produced by this script:
#   - egstalker-base        : Common dependency layer (PyTorch, CUDA libs, pytorch3d, etc.)
#   - egstalker-preprocess  : For preprocessing/face-tracking jobs
#   - egstalker-train       : For training jobs
#   - egstalker-infer       : For prediction/inference jobs
#
# Optional overrides (env vars):
#   TORCH_CUDA_ARCH_LIST="8.6;8.9"  # CUDA arch list passed to Docker build
#   MAX_JOBS=12                     # Parallel compilation jobs for pytorch3d build
#   BASE_IMAGE=egstalker-base       # Base image tag (default: egstalker-base)
#   PREPROCESS_IMAGE=egstalker-preprocess
#   TRAIN_IMAGE=egstalker-train
#   INFER_IMAGE=egstalker-infer
#   VERBOSE=1                       # Show plain Docker build output

# Set progress mode (plain for verbose, auto for quiet)
PROGRESS="${VERBOSE:+plain}"
PROGRESS="${PROGRESS:-auto}"

# Resolve tags
BASE_IMAGE_TAG="${BASE_IMAGE:-egstalker-base}:latest"
PREPROCESS_IMAGE_TAG="${PREPROCESS_IMAGE:-egstalker-preprocess}:latest"
TRAIN_IMAGE_TAG="${TRAIN_IMAGE:-egstalker-train}:latest"
INFER_IMAGE_TAG="${INFER_IMAGE:-egstalker-infer}:latest"

# Base build args
BASE_BUILD_ARGS=()
[[ -n "$TORCH_CUDA_ARCH_LIST" ]] && BASE_BUILD_ARGS+=(--build-arg "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST")
[[ -n "$MAX_JOBS" ]] && BASE_BUILD_ARGS+=(--build-arg "MAX_JOBS=$MAX_JOBS")

echo "[egstalker] Building base image: $BASE_IMAGE_TAG"
docker build \
    --progress="$PROGRESS" \
    -f "$SCRIPT_DIR/Dockerfile.base" \
    -t "$BASE_IMAGE_TAG" \
    "${BASE_BUILD_ARGS[@]}" \
    "$SCRIPT_DIR"

echo "[egstalker] Building preprocess image: $PREPROCESS_IMAGE_TAG"
docker build \
    --progress="$PROGRESS" \
    -f "$SCRIPT_DIR/Dockerfile.preprocess" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE_TAG" \
    -t "$PREPROCESS_IMAGE_TAG" \
    "$SCRIPT_DIR"

echo "[egstalker] Building training image: $TRAIN_IMAGE_TAG"
docker build \
    --progress="$PROGRESS" \
    -f "$SCRIPT_DIR/Dockerfile.train" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE_TAG" \
    -t "$TRAIN_IMAGE_TAG" \
    "$SCRIPT_DIR"

echo "[egstalker] Building inference image: $INFER_IMAGE_TAG"
docker build \
    --progress="$PROGRESS" \
    -f "$SCRIPT_DIR/Dockerfile.infer" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE_TAG" \
    -t "$INFER_IMAGE_TAG" \
    "$SCRIPT_DIR"
