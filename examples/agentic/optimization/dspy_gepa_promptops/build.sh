#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t dspy_gepa_promptops "$SCRIPT_DIR"
