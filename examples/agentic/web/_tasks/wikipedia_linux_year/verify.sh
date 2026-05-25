#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: verify.sh <output-artifacts-dir>}"

# All three artifact types should exist.
[ -f "$OUT/agent_history.json" ] || { echo "FAIL: $OUT/agent_history.json missing"; exit 1; }
[ -f "$OUT/report.md" ]          || { echo "FAIL: $OUT/report.md missing"; exit 1; }

shot_count=$(find "$OUT" -maxdepth 1 -name 'step_*.png' | wc -l)
if [ "$shot_count" -lt 1 ]; then
  echo "FAIL: no step_*.png screenshots saved in $OUT"
  exit 1
fi

# Agent's final result must contain "1991" (the year Linux was first
# released, a stable historical fact). We grep the dedicated section
# of report.md rather than the whole agent log so we don't false-pass
# on a chance "1991" elsewhere in the thinking trace.
final=$(awk '/^## Final result/{flag=1; next} /^## /{flag=0} flag' "$OUT/report.md")
if ! echo "$final" | grep -q "1991"; then
  echo "FAIL: '1991' not in '## Final result' section of $OUT/report.md"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi

echo "ok: $shot_count screenshots + agent_history.json + report.md + final contains '1991'"
