#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t analytical_dowhy_policy "$SCRIPT_DIR"

docker tag analytical_dowhy_policy dowhy_policy_uplift
