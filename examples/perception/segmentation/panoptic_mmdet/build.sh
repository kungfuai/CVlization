#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build from the script's directory, works from anywhere.
# Keep both tags for compatibility with cvl example resolution and older docs.
PRIMARY_TAG="${CVL_IMAGE:-panoptic_mmdet}"
docker build -t "${PRIMARY_TAG}" "$SCRIPT_DIR"
docker tag "${PRIMARY_TAG}" mmdet_ps
