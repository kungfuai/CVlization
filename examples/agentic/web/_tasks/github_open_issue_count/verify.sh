#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: verify.sh <output-artifacts-dir>}"

[ -f "$OUT/agent_history.json" ] || { echo "FAIL: agent_history.json missing"; exit 1; }
[ -f "$OUT/report.md" ]          || { echo "FAIL: report.md missing"; exit 1; }
[ "$(find "$OUT" -maxdepth 1 -name 'step_*.png' | wc -l)" -ge 1 ] \
  || { echo "FAIL: no step_*.png screenshots"; exit 1; }

final=$(awk '/^## Final result/{flag=1; next} /^## /{flag=0} flag' "$OUT/report.md")

# Pick the first integer in the final result (strip commas the model
# might emit despite the prompt).
num=$(echo "$final" | tr -d ',' | grep -oE '\b[0-9]+\b' | head -1)
if [ -z "$num" ]; then
  echo "FAIL: no integer in final result"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi

# Plausibility range: browser-use is a popular repo with hundreds of open
# issues. Reject obviously wrong reads (e.g. extracted version number or
# a year).
if [ "$num" -lt 50 ] || [ "$num" -gt 50000 ]; then
  echo "FAIL: agent reported $num open issues -- outside plausible [50, 50000]"
  exit 1
fi

# Ground-truth via the GitHub search API, scoped to is:issue (excludes
# PRs). The repo's UI "Open" tab on /issues shows this number. The
# alternative /repos endpoint's `open_issues_count` includes PRs and
# will mismatch the agent's read by a lot -- don't use it here.
truth=$(curl -sS "https://api.github.com/search/issues?q=repo:browser-use/browser-use+is:issue+is:open&per_page=1" \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('total_count',0))" \
  2>/dev/null || echo 0)
if [ "$truth" -gt 0 ]; then
  python3 -c "
truth = $truth
got = $num
# Allow 50% drift either way: open-issue counts change daily as people
# file and close. Browser-use closed ~3 issues during the time it took
# us to write this task -- not worth a tight assertion.
ratio = max(truth, got) / max(min(truth, got), 1)
if ratio > 1.5:
    print(f'FAIL: agent reported {got}, API truth {truth} (issues only), ratio {ratio:.2f}x > 1.5x')
    raise SystemExit(1)
print(f'ok: agent={got}, API truth={truth} (issues only), ratio {ratio:.2f}x within tolerance')
"
else
  echo "ok: agent=$num (couldn't ground-truth via API; sanity range OK)"
fi
