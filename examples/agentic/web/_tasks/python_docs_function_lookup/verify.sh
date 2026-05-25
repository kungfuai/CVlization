#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: verify.sh <output-artifacts-dir>}"

[ -f "$OUT/agent_history.json" ] || { echo "FAIL: agent_history.json missing"; exit 1; }
[ -f "$OUT/report.md" ]          || { echo "FAIL: report.md missing"; exit 1; }
[ "$(find "$OUT" -maxdepth 1 -name 'step_*.png' | wc -l)" -ge 1 ] \
  || { echo "FAIL: no step_*.png screenshots"; exit 1; }

final=$(awk '/^## Final result/{flag=1; next} /^## /{flag=0} flag' "$OUT/report.md")

# The function that returns the current working directory is os.getcwd().
# (There's also os.getcwdb() which returns bytes -- accept either since
# both are technically correct answers to the question.)
if ! echo "$final" | grep -qE '\b(os\.)?getcwd[b]?\b'; then
  echo "FAIL: final result doesn't contain 'getcwd' (or 'os.getcwd')"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi

# Negative: should not contain getcwdu (deprecated Python 2 alias) -- if
# the agent reaches for that, it's pulling from training data, not the
# actual current docs.
if echo "$final" | grep -qE '\bos\.getcwdu\b'; then
  echo "FAIL: 'os.getcwdu' is Python 2 only -- agent pulled stale info, not from docs"
  exit 1
fi

echo "ok: agent extracted 'getcwd' from the os module docs"
