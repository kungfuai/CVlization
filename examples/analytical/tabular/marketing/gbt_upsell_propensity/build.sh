#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t analytical_gbt_upsell "$SCRIPT_DIR"

# Tag image for CVL preset compatibility (expects gbt_upsell_propensity)
docker tag analytical_gbt_upsell gbt_upsell_propensity
