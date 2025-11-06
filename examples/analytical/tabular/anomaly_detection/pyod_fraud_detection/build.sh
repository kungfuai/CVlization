#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t analytical_pyod_fraud "$SCRIPT_DIR"

docker tag analytical_pyod_fraud pyod_fraud_detection
