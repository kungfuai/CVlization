#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t cvlization/vlm-omr-miles:latest "$SCRIPT_DIR"
echo "Build complete: cvlization/vlm-omr-miles:latest"
