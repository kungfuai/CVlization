#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t analytical_pycox_retention "$SCRIPT_DIR"

docker tag analytical_pycox_retention pycox_retention
