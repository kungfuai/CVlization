#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t pycaret-structured "$SCRIPT_DIR"

docker tag pycaret-structured pycaret_structured
