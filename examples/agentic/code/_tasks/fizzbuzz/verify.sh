#!/usr/bin/env bash
set -euo pipefail
WORK="${1:?usage: verify.sh <scratch-dir>}"
cd "$WORK"

# Two files must exist.
[ -f fizzbuzz.py ] || { echo "FAIL: fizzbuzz.py missing"; exit 1; }
[ -f test_fizzbuzz.py ] || { echo "FAIL: test_fizzbuzz.py missing"; exit 1; }

# pytest must collect 4 cases and pass them all. Re-run via python -m pytest so
# we use whatever pytest is on PATH (host venv, not the agent container).
python3 -m pytest -q test_fizzbuzz.py
