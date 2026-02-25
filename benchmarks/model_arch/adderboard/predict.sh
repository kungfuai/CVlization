#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <submission.py> [verify.py args...]"
  exit 1
fi

SUBMISSION="$1"
shift

python verify.py "$SUBMISSION" "$@"
