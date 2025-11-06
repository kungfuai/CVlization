#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t analytical_mapie_conformal "$SCRIPT_DIR"

docker tag analytical_mapie_conformal mapie_conformal
