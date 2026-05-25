#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: verify.sh <output-artifacts-dir>}"

[ -f "$OUT/agent_history.json" ] || { echo "FAIL: agent_history.json missing"; exit 1; }
[ -f "$OUT/report.md" ]          || { echo "FAIL: report.md missing"; exit 1; }
shot_count=$(find "$OUT" -maxdepth 1 -name 'step_*.png' | wc -l)
[ "$shot_count" -ge 1 ] || { echo "FAIL: no step_*.png screenshots"; exit 1; }

# Pull the agent's final result from the report. It's wrapped in ``` fences,
# strip them so we can grep the markdown structure beneath.
final=$(awk '/^## Final result/{flag=1; next} /^## /{flag=0} flag' "$OUT/report.md" \
        | sed -e 's/^```$//' -e 's/^```.*$//')

# --- Structure checks --------------------------------------------------------

# Length: the brief should be substantive but not a wall of text.
chars=$(echo -n "$final" | wc -c)
if [ "$chars" -lt 400 ]; then
  echo "FAIL: brief is only $chars chars; expected >=400 for a real research brief"
  exit 1
fi
if [ "$chars" -gt 5000 ]; then
  echo "FAIL: brief is $chars chars; expected <=5000 (looks like the agent dumped extra)"
  exit 1
fi

# Required H1 heading (the prompt asks for it verbatim, but accept minor
# variation in dashes / spaces / case).
if ! echo "$final" | grep -qiE '^#[[:space:]]+the foundations of modern llms'; then
  echo "FAIL: missing required H1 'The Foundations of Modern LLMs ...'"
  exit 1
fi

# Markdown table: header pipes + delimiter row
if ! echo "$final" | grep -q '| Model'; then
  echo "FAIL: missing markdown table header containing 'Model'"
  exit 1
fi
if ! echo "$final" | grep -qE '\|[[:space:]]*-{2,}'; then
  echo "FAIL: missing markdown table delimiter row (| --- |)"
  exit 1
fi

# Common themes subheading.
if ! echo "$final" | grep -qiE '^##[[:space:]]+common themes'; then
  echo "FAIL: missing '## Common themes' subheading"
  exit 1
fi

# --- Content checks: known facts the agent must have actually extracted ----

# Transformer was introduced in 2017 by Google. BERT in 2018 also by Google.
# GPT-3 in 2020 by OpenAI. The brief should mention each.
required=(
  "Transformer"
  "BERT"
  "GPT-3"
  "2017"          # Transformer year
  "2018"          # BERT year
  "2020"          # GPT-3 year
  "Google"        # Transformer + BERT org
  "OpenAI"        # GPT-3 org
)
missing=()
for term in "${required[@]}"; do
  if ! echo "$final" | grep -qi "$term"; then
    missing+=("$term")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "FAIL: brief is missing required facts: ${missing[*]}"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi

# GPT-3's parameter count (175B / 175 billion) is one of its most-cited
# facts; the brief should mention it (the agent's "parameters" column should
# have this for GPT-3).
if ! echo "$final" | grep -qiE '175[[:space:]]*(b|billion)'; then
  echo "FAIL: brief doesn't mention GPT-3's '175B' / '175 billion' parameter count"
  exit 1
fi

echo "ok: $shot_count screenshots + $chars-char brief contains all required structure + facts"
