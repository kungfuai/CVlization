#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build from the script's directory, works from anywhere
docker build --pull --no-cache -t mixtral8x7b "$SCRIPT_DIR"
