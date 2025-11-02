#!/usr/bin/env bash
set -euo pipefail

# Build the Docker image from this example directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t cifar10_speedrun "$SCRIPT_DIR"
