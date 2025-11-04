#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t analytical_gbt_housing "$SCRIPT_DIR"

docker tag analytical_gbt_housing gbt_housing_prices
