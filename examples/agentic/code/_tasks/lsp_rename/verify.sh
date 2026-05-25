#!/usr/bin/env bash
set -euo pipefail
WORK="${1:?usage: verify.sh <scratch-dir>}"
cd "$WORK"

# After the rename: zero references to the old name, multiple to the new.
if grep -rn "add_numbers" src/ >/dev/null 2>&1; then
  echo "FAIL: 'add_numbers' still present after rename:"
  grep -rn "add_numbers" src/
  exit 1
fi

# Expected sites: def in lib.py, import in main.py, call in main.py,
# import in util.py, call in util.py = 5 occurrences of sum_values.
count=$(grep -rn "sum_values" src/ | wc -l)
if [ "$count" -lt 5 ]; then
  echo "FAIL: expected >= 5 occurrences of 'sum_values', got $count:"
  grep -rn "sum_values" src/ || true
  exit 1
fi

# def site is in lib.py
grep -q "^def sum_values" src/lib.py || { echo "FAIL: def sum_values missing from lib.py"; exit 1; }

# Project must still import-OK from cwd (the rename didn't accidentally break syntax)
python3 -c "import sys; sys.path.insert(0, '.'); import src.lib, src.main, src.util" \
  || { echo "FAIL: post-rename modules don't import"; exit 1; }

echo "ok: $count occurrences of sum_values across src/"
