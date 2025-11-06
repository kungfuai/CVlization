#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t autogluon-structured "$SCRIPT_DIR"

docker tag autogluon-structured autogluon_structured
