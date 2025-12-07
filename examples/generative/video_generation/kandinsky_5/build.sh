#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t cvlization/kandinsky-5:latest "${SCRIPT_DIR}"
