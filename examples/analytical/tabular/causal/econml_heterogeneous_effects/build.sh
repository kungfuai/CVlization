#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t econml_heterogeneous_effects "$SCRIPT_DIR"
docker tag econml_heterogeneous_effects econml-heterogeneous-effects
