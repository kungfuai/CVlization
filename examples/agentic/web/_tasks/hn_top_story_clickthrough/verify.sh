#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: verify.sh <output-artifacts-dir>}"

[ -f "$OUT/agent_history.json" ] || { echo "FAIL: agent_history.json missing"; exit 1; }
[ -f "$OUT/report.md" ]          || { echo "FAIL: report.md missing"; exit 1; }
[ "$(find "$OUT" -maxdepth 1 -name 'step_*.png' | wc -l)" -ge 1 ] \
  || { echo "FAIL: no step_*.png screenshots"; exit 1; }

final=$(awk '/^## Final result/{flag=1; next} /^## /{flag=0} flag' "$OUT/report.md")

# Extract the first http(s) URL the agent produced.
url=$(echo "$final" | grep -oE 'https?://[^[:space:]<>"`]+' | head -1)
if [ -z "$url" ]; then
  echo "FAIL: no URL in final result"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi

# Reject the HN homepage itself (the agent must have clicked through).
# Accept news.ycombinator.com/item?id=... since Show HN / Ask HN / jobs
# legitimately stay on HN.
if echo "$url" | grep -qE '^https?://news\.ycombinator\.com/?(news)?(\?p=[0-9]+)?$'; then
  echo "FAIL: agent reported HN homepage URL; expected destination of #1 story"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi

# Sanity: URL has a real host with at least one dot.
host=$(python3 -c "from urllib.parse import urlparse; print(urlparse('$url').netloc)")
if ! echo "$host" | grep -q '\.'; then
  echo "FAIL: URL host doesn't look real: '$host'"
  exit 1
fi

echo "ok: agent clicked through to destination URL: $url"
