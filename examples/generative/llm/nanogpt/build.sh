#!/bin/bash
# Works from both repo root and example directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build -t nanogpt "$SCRIPT_DIR"
