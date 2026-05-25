#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: verify.sh <output-artifacts-dir>}"

[ -f "$OUT/agent_history.json" ] || { echo "FAIL: agent_history.json missing"; exit 1; }
[ -f "$OUT/report.md" ]          || { echo "FAIL: report.md missing"; exit 1; }
[ "$(find "$OUT" -maxdepth 1 -name 'step_*.png' | wc -l)" -ge 1 ] \
  || { echo "FAIL: no step_*.png screenshots"; exit 1; }

final=$(awk '/^## Final result/{flag=1; next} /^## /{flag=0} flag' "$OUT/report.md")

# Paper is "Attention Is All You Need" -- the canonical Transformer paper.
# This is the title both in the PDF and on the abs page, stable forever
# (arxiv IDs are immutable). Case-insensitive substring match accommodates
# the agent dropping or capitalising 'Is'/'All'/'You' inconsistently.
if ! echo "$final" | grep -iqE 'attention[[:space:]]+is[[:space:]]+all[[:space:]]+you[[:space:]]+need'; then
  echo "FAIL: final result doesn't contain 'Attention Is All You Need'"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi

echo "ok: agent extracted 'Attention Is All You Need' from the PDF"
