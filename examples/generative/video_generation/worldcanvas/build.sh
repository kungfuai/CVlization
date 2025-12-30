#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build -t worldcanvas "$SCRIPT_DIR"
