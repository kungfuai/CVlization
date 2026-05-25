#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: verify.sh <output-artifacts-dir>}"

[ -f "$OUT/agent_history.json" ] || { echo "FAIL: agent_history.json missing"; exit 1; }
[ -f "$OUT/report.md" ]          || { echo "FAIL: report.md missing"; exit 1; }
[ "$(find "$OUT" -maxdepth 1 -name 'step_*.png' | wc -l)" -ge 1 ] \
  || { echo "FAIL: no step_*.png screenshots"; exit 1; }

final=$(awk '/^## Final result/{flag=1; next} /^## /{flag=0} flag' "$OUT/report.md")

# Forgiving acceptance: the agent must have produced a 'match:' or
# 'mismatch:' verdict line WITH at least one pytest-shaped semver
# (8.X.Y). Either verdict counts as PASS as long as the agent actually
# attempted the comparison and reported a version. We don't assert the
# verdict's correctness because the agent might mis-read either source;
# the task is a known 5+ hop stretch for small VLMs (see README).
verdict=$(echo "$final" | grep -iE '^[[:space:]]*(match|mismatch):' | head -1 || true)
if [ -z "$verdict" ]; then
  echo "FAIL: no 'match:' or 'mismatch:' line in final result -- agent didn't complete the comparison"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi

# At least one pytest-shaped version (8.X or 8.X.Y) somewhere in verdict.
if ! echo "$verdict" | grep -qE '\b8\.[0-9]+(\.[0-9]+)?\b'; then
  echo "FAIL: verdict line missing a pytest-shaped version (expected 8.X.Y)"
  echo "  verdict: $verdict"
  exit 1
fi

echo "ok: agent produced verdict line: $verdict"
