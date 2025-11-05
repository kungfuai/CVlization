#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

docker build -t moirai_zero_shot "$SCRIPT_DIR"
docker tag moirai_zero_shot moirai-zero-shot
