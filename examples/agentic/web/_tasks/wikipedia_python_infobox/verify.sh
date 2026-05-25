#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: verify.sh <output-artifacts-dir>}"

[ -f "$OUT/agent_history.json" ] || { echo "FAIL: agent_history.json missing"; exit 1; }
[ -f "$OUT/report.md" ]          || { echo "FAIL: report.md missing"; exit 1; }
[ "$(find "$OUT" -maxdepth 1 -name 'step_*.png' | wc -l)" -ge 1 ] \
  || { echo "FAIL: no step_*.png screenshots"; exit 1; }

final=$(awk '/^## Final result/{flag=1; next} /^## /{flag=0} flag' "$OUT/report.md")

# Find the first balanced JSON object in the final result. report.md wraps
# it in ``` fences, so strip those first.
json=$(echo "$final" | sed 's/^```.*$//' | tr -d '\r' | python3 -c "
import re, sys
text = sys.stdin.read()
m = re.search(r'\{[^{}]*\}', text)
if m: print(m.group(0))
")

if [ -z "$json" ]; then
  echo "FAIL: no JSON object in final result"
  echo "--- got: ---"
  echo "$final"
  exit 1
fi

python3 <<PY
import json, sys
raw = '''$json'''
try:
    d = json.loads(raw)
except Exception as e:
    print(f"FAIL: JSON didn't parse: {e}")
    print(f"  raw: {raw!r}")
    sys.exit(1)

missing = [k for k in ("first_appeared", "paradigm", "typing_discipline") if k not in d]
if missing:
    print(f"FAIL: missing keys {missing}")
    sys.exit(1)

# first_appeared should be 1991 (Python's first release year; stable forever).
fa = str(d["first_appeared"]).strip()
if fa != "1991":
    print(f"FAIL: first_appeared expected '1991', got {fa!r}")
    sys.exit(1)

# paradigm should mention object-orientation (a fact about Python that's
# stable in any reasonable infobox).
para = str(d["paradigm"]).lower()
if "object" not in para:
    print(f"FAIL: paradigm doesn't mention 'object': {para!r}")
    sys.exit(1)

# typing_discipline should be non-trivial.
if len(str(d["typing_discipline"]).strip()) < 3:
    print(f"FAIL: typing_discipline too short: {d['typing_discipline']!r}")
    sys.exit(1)

print(f"ok: JSON parsed; first_appeared=1991; paradigm mentions 'object'; typing_discipline non-trivial")
PY
