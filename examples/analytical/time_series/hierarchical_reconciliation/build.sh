#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t hierarchical_reconciliation "$SCRIPT_DIR"
docker tag hierarchical_reconciliation hierarchical-reconciliation
