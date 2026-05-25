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

# Extract the '## Final result' block of the report. The httpbin response's
# 'url' field is exactly 'https://httpbin.org/post' -- different from the
# /forms/post page the agent started on. Distinguishing /post from
# /forms/post proves the agent actually submitted and read the response,
# rather than regurgitating the URL from the prompt.
final=$(awk '/^## Final result/{flag=1; next} /^## /{flag=0} flag' "$OUT/report.md")

if ! echo "$final" | grep -qE 'httpbin\.org/post([^/]|$)'; then
  echo "FAIL: agent's final result doesn't contain 'httpbin.org/post' (without /forms/)"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi

# Negative assertion: result should NOT be the original /forms/post URL.
if echo "$final" | grep -qE 'httpbin\.org/forms/post'; then
  echo "FAIL: result still references /forms/post -- agent likely didn't submit"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi

echo "ok: $shot_count screenshots + artifacts + final result correctly = https://httpbin.org/post"
