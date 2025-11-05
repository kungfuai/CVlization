#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t analytical_ranking_lightgbm "$SCRIPT_DIR"
docker tag analytical_ranking_lightgbm ranking_lightgbm
