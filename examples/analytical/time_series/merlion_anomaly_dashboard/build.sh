#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t merlion_anomaly_dashboard "$SCRIPT_DIR"
docker tag merlion_anomaly_dashboard merlion-anomaly-dashboard
