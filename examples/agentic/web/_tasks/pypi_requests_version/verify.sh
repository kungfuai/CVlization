#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: verify.sh <output-artifacts-dir>}"

[ -f "$OUT/agent_history.json" ] || { echo "FAIL: agent_history.json missing"; exit 1; }
[ -f "$OUT/report.md" ]          || { echo "FAIL: report.md missing"; exit 1; }

shot_count=$(find "$OUT" -maxdepth 1 -name 'step_*.png' | wc -l)
if [ "$shot_count" -lt 1 ]; then
  echo "FAIL: no step_*.png in $OUT"
  exit 1
fi

final=$(awk '/^## Final result/{flag=1; next} /^## /{flag=0} flag' "$OUT/report.md")

# Find a semver-like number in the final result.
ver=$(echo "$final" | grep -oE '\b2\.[0-9]+\.[0-9]+\b' | head -1 || true)
if [ -z "$ver" ]; then
  echo "FAIL: no '2.X.Y' version string in final result"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi

# Floor check: latest as of 2026-05 is 2.32.5; assert minor >= 28 to allow
# some lag and rollback room without false-passing on totally stale numbers.
minor=$(echo "$ver" | cut -d. -f2)
if [ "$minor" -lt 28 ]; then
  echo "FAIL: extracted version $ver but minor < 28 (latest is in the 2.32.x range)"
  exit 1
fi

echo "ok: $shot_count screenshots + artifacts + extracted version $ver"
