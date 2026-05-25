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

# torvalds/linux (~234k stars) vs microsoft/typescript (~109k stars) -- a
# 2.15x ratio and ~125k absolute gap as of 2026-05. The kernel is a Linus
# Torvalds personal repo on GitHub since 2012; TypeScript is a Microsoft
# language project. Their relative ranking has been stable for years and
# the gap is wide enough that it should remain stable. If this assertion
# starts failing for "agent correctly identified the bigger repo" reasons,
# check the current numbers via:
#   curl -sS https://api.github.com/repos/torvalds/linux | jq .stargazers_count
#   curl -sS https://api.github.com/repos/microsoft/typescript | jq .stargazers_count
if ! echo "$final" | grep -iqE 'torvalds/linux|^\s*linux\s*$'; then
  echo "FAIL: agent's final result doesn't name 'linux' as the winner"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi
# Negative: the answer line should not BE typescript.
last_line=$(echo "$final" | grep -v '^$' | grep -v '^```' | tail -1 | tr -d ' \t\r')
if [ "$last_line" = "microsoft/typescript" ] || [ "$last_line" = "typescript" ]; then
  echo "FAIL: agent's final answer is typescript (wrong; linux has more stars)"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi

echo "ok: $shot_count screenshots + artifacts + agent named torvalds/linux as winner"
