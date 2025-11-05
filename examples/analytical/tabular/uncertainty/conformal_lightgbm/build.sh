#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t analytical_conformal_lightgbm "$SCRIPT_DIR"
docker tag analytical_conformal_lightgbm conformal_lightgbm
