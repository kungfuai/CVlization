#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t dowhy_berkeley_bias "$SCRIPT_DIR"
docker tag dowhy_berkeley_bias dowhy-berkeley-bias
