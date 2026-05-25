#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: verify.sh <output-artifacts-dir>}"

[ -f "$OUT/agent_history.json" ] || { echo "FAIL: agent_history.json missing"; exit 1; }
[ -f "$OUT/report.md" ]          || { echo "FAIL: report.md missing"; exit 1; }
shot_count=$(find "$OUT" -maxdepth 1 -name 'step_*.png' | wc -l)
[ "$shot_count" -ge 1 ] || { echo "FAIL: no step_*.png screenshots"; exit 1; }

# Pull the agent's final result from the report. With BROWSER_USE_OUTPUT_MODEL
# the final result is a JSON object matching the ResearchBrief schema.
final=$(awk '/^## Final result/{flag=1; next} /^## /{flag=0} flag' "$OUT/report.md" \
        | sed -e 's/^```$//' -e 's/^```.*$//')

# Parse + validate via Python.
python3 <<PY
import json, re, sys

final = """$final"""

# Strip leading/trailing whitespace; find first balanced { ... } object.
m = re.search(r'\{.*\}', final, re.DOTALL)
if not m:
    print("FAIL: no JSON object in final result")
    print("--- got: ---")
    print(final[:800])
    sys.exit(1)

raw = m.group(0)
try:
    d = json.loads(raw)
except Exception as e:
    print(f"FAIL: JSON didn't parse: {e}")
    print(f"--- raw: ---\n{raw[:800]}")
    sys.exit(1)

# Required fields.
missing = [k for k in ("title", "introduction", "table_rows", "common_themes") if k not in d]
if missing:
    print(f"FAIL: missing top-level fields: {missing}")
    sys.exit(1)

# table_rows: list of >=3 dicts with model/year/org/parameters.
rows = d["table_rows"]
if not isinstance(rows, list) or len(rows) < 3:
    print(f"FAIL: table_rows expected >=3 rows, got {len(rows) if isinstance(rows, list) else type(rows).__name__}")
    sys.exit(1)
row_keys = {"model", "year", "org", "parameters"}
for i, r in enumerate(rows[:3]):
    if not isinstance(r, dict):
        print(f"FAIL: table_rows[{i}] not a dict")
        sys.exit(1)
    missing_r = row_keys - set(r.keys())
    if missing_r:
        print(f"FAIL: table_rows[{i}] missing keys: {missing_r}")
        sys.exit(1)

# common_themes: required, non-empty, >=50 chars.
themes = d["common_themes"]
if not isinstance(themes, str) or len(themes.strip()) < 50:
    print(f"FAIL: common_themes too short ({len(str(themes))} chars; need >=50)")
    print(f"  themes: {themes!r}")
    sys.exit(1)

# Required facts: 2017 Transformer, 2018 BERT, 2020 GPT-3, Google, OpenAI,
# and GPT-3's 175B parameter count must appear somewhere in the JSON (we
# stringify the whole brief and grep).
blob = json.dumps(d).lower()
required_facts = ["transformer", "bert", "gpt-3", "2017", "2018", "2020",
                  "google", "openai"]
missing_f = [f for f in required_facts if f not in blob]
if missing_f:
    print(f"FAIL: brief missing required facts: {missing_f}")
    sys.exit(1)
# GPT-3 175B parameter count
if not re.search(r'175\s*(b|billion)', blob):
    print(f"FAIL: missing GPT-3 175B / 175 billion parameter count")
    sys.exit(1)

print(f"ok: ResearchBrief parsed; "
      f"{len(rows)} table rows; common_themes={len(themes)} chars; all key facts present")
PY
