#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t causalml_campaign_optimization "$SCRIPT_DIR"
docker tag causalml_campaign_optimization causalml-campaign-optimization
