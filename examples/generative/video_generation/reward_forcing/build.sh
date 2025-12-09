#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CVL CLI uses underscore tags (see cvl run output: Docker: reward_forcing)
IMAGE="${CVL_IMAGE:-reward_forcing:latest}"

docker build -t "${IMAGE}" "${SCRIPT_DIR}"
