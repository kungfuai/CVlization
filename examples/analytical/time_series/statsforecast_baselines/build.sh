#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t statsforecast_baselines "$SCRIPT_DIR"
docker tag statsforecast_baselines statsforecast-baselines
