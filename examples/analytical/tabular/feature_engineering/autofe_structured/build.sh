#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t analytical_autofe_structured "$SCRIPT_DIR"

docker tag analytical_autofe_structured autofe_structured
