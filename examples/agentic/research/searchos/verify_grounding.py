"""Strict offline grounding gate for a SearchOS run (stdlib only, host-run).

Checks a run's ``search_state.json`` against the offline corpus and enforces,
for **every filled non-identity attribute cell** of the coverage map:

  1. at least one active evidence node for that (table, entity, attribute)
     whose ``source`` is an http(s) URL that exists in ``corpus.json``;
  2. that node's ``source_excerpt`` is a non-empty **verbatim substring** of
     the source page's content;
  3. the cell's value actually appears in that source page (case-insensitive,
     whitespace-normalized), so the citation is inspectable, not decorative.
     Exception: when the cell's value is itself an http(s) URL (the LLM
     sometimes designs explicit ``*_citation`` provenance columns), the cell
     is anchored iff its anchored evidence comes from exactly that URL — the
     claimed citation must match the actual page the excerpt was taken from.

Evidence with ``agent://`` sources (explore/summary replay, orchestrator
assertion) never satisfies the gate — such cells count as UNRESOLVED even
when their values happen to be correct. Identity cells (primary-key columns,
or cells whose value is the row's own entity name) are exempt: their "value"
is the row identity established at schema time, not a researched fact.

Usage:
  python verify_grounding.py [--state searchos_outputs/search_state.json]
                             [--corpus corpus.json] [--require-complete]

Exit 0: every filled non-identity cell is page-anchored (with
``--require-complete``, additionally no cell may be unfilled).
Exit 1: any filled cell lacks anchored corpus evidence (or, with
``--require-complete``, any cell is missing).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _value_in_page(value: str, page: str) -> bool:
    value_n, page_n = _norm(value), _norm(page)
    if not value_n:
        return False
    if value_n in page_n:
        return True
    # Tolerate list-ish / annotated cell values ("Tallinn, Estonia" vs page
    # "in Tallinn, Estonia."): require every comma-separated part in the page.
    parts = [p for p in (_norm(p) for p in re.split(r"[;,]", value)) if p]
    return bool(parts) and all(p in page_n for p in parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", default=str(HERE / "searchos_outputs" / "search_state.json"))
    ap.add_argument("--corpus", default=str(HERE / "corpus.json"))
    ap.add_argument("--require-complete", action="store_true",
                    help="also fail if any coverage cell is unfilled")
    args = ap.parse_args(argv)

    corpus = {
        d["url"]: d.get("content", "")
        for d in json.loads(Path(args.corpus).read_text(encoding="utf-8"))["documents"]
    }
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))

    nodes = state.get("evidence_graph", {}).get("nodes", {}) or {}
    node_list = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)

    cov = state.get("coverage_map", {}) or {}
    tables = cov.get("tables", {}) or {}
    cells = cov.get("cells", {}) or {}

    anchored = unresolved = missing = identity = 0
    rows: list[str] = []

    for tid, tbl in tables.items():
        pk_cols = set(tbl.get("primary_key") or [])
        for entity in tbl.get("entities", []) or []:
            for attr in tbl.get("attributes", []) or []:
                cell = cells.get(f"{tid}/{entity}.{attr}", {}) or {}
                value = str(cell.get("value") or "")
                filled = cell.get("status") == "filled"
                if attr in pk_cols or (filled and _norm(value) == _norm(entity)):
                    identity += 1
                    continue
                if not filled:
                    missing += 1
                    rows.append(f"MISSING    | {entity} · {attr}")
                    continue
                proof = None
                for n in node_list:
                    if n.get("status") not in (None, "", "active"):
                        continue
                    if (n.get("table_id") or tid) != tid:
                        continue
                    if _norm(n.get("entity", "")) != _norm(entity):
                        continue
                    if _norm(n.get("attribute", "")) != _norm(attr):
                        continue
                    src = n.get("source", "") or ""
                    page = corpus.get(src)
                    if page is None or not src.startswith(("http://", "https://")):
                        continue  # agent:// or unknown source — never counts
                    excerpt = (n.get("source_excerpt") or "").strip()
                    if not excerpt or excerpt not in page:
                        continue
                    if value.startswith(("http://", "https://")):
                        # Provenance column: the claimed citation URL must be
                        # the very page this anchored evidence came from.
                        if _norm(value) != _norm(src):
                            continue
                    elif not _value_in_page(value, page):
                        continue
                    proof = src
                    break
                if proof:
                    anchored += 1
                    rows.append(f"ANCHORED   | {entity} · {attr} = {value!r} ← {proof}")
                else:
                    unresolved += 1
                    rows.append(f"UNRESOLVED | {entity} · {attr} = {value!r} "
                                "(no anchored corpus evidence)")

    print("\n".join(rows))
    print(f"\nanchored={anchored}  unresolved={unresolved}  missing={missing}  "
          f"identity={identity}")

    ok = unresolved == 0 and (missing == 0 or not args.require_complete)
    print("GROUNDING GATE:", "PASS" if ok else "FAIL",
          "(strict: every filled non-identity cell is page-anchored"
          + (", coverage complete)" if args.require_complete else ")"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
