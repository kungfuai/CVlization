"""Thin CVlization driver for a single SearchOS open-domain research run.

SearchOS (https://github.com/antins-labs/SearchOS, arXiv:2607.15257) compiles an
open-domain question into a normalized coverage map, dispatches its empty cells
to pipelined-parallel sub-agents, writes every fact — with its source — into a
shared evidence graph (SOCM), and synthesizes a citation-grounded answer.

This wrapper runs *one* such session end-to-end and collects the artifacts a
CVlization example should surface:

  * ``report.md``        — the synthesized, citation-grounded answer
  * ``search_state.json``— the full SOCM state (coverage map + evidence graph +
                           frontier task queue + strategy/failure memory)
  * ``summary.json``     — verdict, coverage %, evidence count, steps, tokens
  * a printed SOCM view  — the coverage map filling in, cell by cell

Two search modes, one identical pipeline:

  --mode offline (default): a keyless :class:`LocalCorpusProvider` serves results
      from ``corpus.json``; the browser backend is set to ``search_engine`` so
      page ``open()`` is served from that corpus with **no network at all**.
      Only an LLM key is needed. This is the reproducible demo/CI path.

  --mode web: the real SearchOS path — Serper or Tavily web search + live page
      fetch. Needs ``SERPER_API_KEY`` or ``TAVILY_API_KEY`` in addition to the
      LLM key.

The LLM is chosen with SearchOS's one-knob ``SF_PROVIDER`` preset (e.g.
``openai``, ``anthropic``, ``deepseek``); ``--model`` / ``--fast-model`` override
the model ids within that provider.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_search.py",
        description="Run one SearchOS open-domain research session and collect artifacts.",
    )
    p.add_argument("query", nargs="?", default=None, help="Research query (open-domain question).")
    p.add_argument("--mode", choices=["offline", "web"], default="offline",
                   help="offline = keyless local-corpus search (default); web = real Serper/Tavily.")
    p.add_argument("--corpus", default=str(HERE / "corpus.json"),
                   help="Path to the offline corpus JSON (offline mode only).")
    p.add_argument("--provider", default=os.environ.get("SF_PROVIDER", "openai"),
                   help="SearchOS SF_PROVIDER preset (openai | anthropic | deepseek | ...).")
    p.add_argument("--model", default=os.environ.get("SF_MODEL", ""),
                   help="Override the provider's main model id (orchestrator / sub-agents).")
    p.add_argument("--fast-model", default=os.environ.get("SF_FAST_MODEL", ""),
                   help="Override the provider's light-tier model id (extraction / synthesis).")
    p.add_argument("--search-provider", default=os.environ.get("SF_SEARCH_PROVIDER", ""),
                   help="web mode: serper | tavily (default: auto-detect from present API key).")
    p.add_argument("--output-dir", default=os.environ.get("SEARCHOS_OUTPUT_DIR", "./searchos_outputs"),
                   help="Where to write the workspace + collected artifacts.")
    p.add_argument("--max-parallel-agents", type=int, default=3,
                   help="Cap on concurrent sub-agents (keeps the demo bounded/cheap).")
    p.add_argument("--max-iterations", type=int, default=24,
                   help="Cap on orchestrator loop iterations.")
    p.add_argument("--max-time-s", type=int, default=600, help="Wall-clock budget (0 = unlimited).")
    p.add_argument("--enable-skills", action="store_true",
                   help="Enable the hierarchical skill system (off by default for a lean demo).")
    return p.parse_args(argv)


DEFAULT_QUERY = (
    "For each AI research lab profiled in the SearchOS demo corpus — Nimbus Labs, "
    "Orchard AI, Basilisk Research, and Tidewater Institute — find its founding year, "
    "headquarters city, and flagship open model. Present the results as a table with a "
    "citation for every cell."
)


def _configure_env(args: argparse.Namespace) -> None:
    """Set SF_* env vars BEFORE any searchos import (settings snapshot at import)."""
    os.environ["SF_PROVIDER"] = args.provider
    if args.model:
        os.environ["SF_MODEL"] = args.model
    if args.fast_model:
        os.environ["SF_FAST_MODEL"] = args.fast_model

    # Bounded, reproducible demo defaults.
    os.environ["SF_MAX_PARALLEL_AGENTS"] = str(args.max_parallel_agents)
    os.environ["SF_ORCH_MAX_ITERATIONS"] = str(args.max_iterations)
    os.environ["SF_DEFAULT_MAX_TIME_S"] = str(args.max_time_s)
    # gpt-4.1-class orchestrators tend to end their turn while sub-agents are
    # still browsing; the harness re-prompts ("premature end resume") up to
    # this many times so in-flight agents can land their anchored evidence.
    os.environ.setdefault("SF_ORCH_PREMATURE_END_MAX_RESUMES", "5")
    os.environ["SF_ENABLE_SKILLS"] = "true" if args.enable_skills else "false"
    os.environ["SF_ENABLE_SKILL_ROUTER"] = "true" if args.enable_skills else "false"

    # Faithful grounding: the Explore scout still discovers entities + hub pages,
    # but its natural-language summary is NOT replayed into the evidence graph.
    # That keeps every filled cell backed by an actual page-open (tier-2 anchored
    # evidence with a real source URL) instead of an unsourced scout guess.
    os.environ["SF_ENABLE_EXPLORE_REPLAY"] = "false"
    # Same for the search sub-agents' final summaries (gate added by this
    # example's build-time patch, see patch_search_persona.py): a hallucinated
    # summary must not fill coverage cells as agent://…/final_summary evidence.
    # With replay off, cells fill only from anchored page-open extraction, and
    # an un-grounded cell stays open for re-dispatch.
    os.environ["SF_ENABLE_SUMMARY_REPLAY"] = "false"

    workspace = str(Path(args.output_dir).resolve() / "workspace")
    os.environ["SF_WORKSPACE_ROOT"] = workspace

    if args.mode == "offline":
        # Serve page open() from the in-memory search-result cache — no network.
        os.environ["SF_BROWSER_BACKEND"] = "search_engine"
    elif args.search_provider:
        os.environ["SF_SEARCH_PROVIDER"] = args.search_provider


def _install_provider(args: argparse.Namespace) -> str:
    """Wire the search backend. Returns a human-readable provider label."""
    from searchos.tools.simple_browser.state import set_browser_provider

    if args.mode == "offline":
        sys.path.insert(0, str(HERE))
        from local_corpus_provider import LocalCorpusProvider

        provider = LocalCorpusProvider.from_json(args.corpus)
        set_browser_provider(provider)
        return f"local-corpus ({Path(args.corpus).name})"

    from searchos.tools.simple_browser.search import (
        build_search_provider,
        resolve_search_provider_name,
    )

    name = resolve_search_provider_name(args.search_provider)
    if name == "ragflow":
        raise SystemExit(
            "web mode needs a web-search API key: set SERPER_API_KEY or "
            "TAVILY_API_KEY (both offer free tiers). ragflow is Ant-internal only."
        )
    set_browser_provider(build_search_provider(name))
    return name


def _render_socm(state: dict) -> str:
    """Compact human view of the SOCM state saved in search_state.json."""
    lines: list[str] = []
    cov = state.get("coverage_map", {}) or {}
    tables = cov.get("tables", {}) or {}
    cells = cov.get("cells", {}) or {}

    def cell_lookup(tid: str, entity: str, attr: str) -> dict:
        return cells.get(f"{tid}/{entity}.{attr}", {}) or {}

    mark = {"filled": "✓", "uncertain": "~", "hard_cell": "✗", "missing": "·"}
    for tid, tbl in tables.items():
        label = tbl.get("table_label") or tid
        entities = tbl.get("entities", []) or []
        attrs = tbl.get("attributes", []) or []
        lines.append(f"  Table `{label}`  ({len(entities)} rows × {len(attrs)} cols)")
        header = "    row".ljust(28) + "  ".join(a[:14].ljust(14) for a in attrs)
        lines.append(header)
        for ent in entities:
            row = f"    {ent[:24]:<24}"
            marks = []
            for a in attrs:
                c = cell_lookup(tid, ent, a)
                marks.append(mark.get(c.get("status", "missing"), "·").ljust(14))
            lines.append(row + "  " + "  ".join(marks))
    filled = sum(1 for c in cells.values() if (c or {}).get("status") == "filled")
    lines.append(f"  cells filled: {filled}/{len(cells)}")

    ev = state.get("evidence_graph", {}) or {}
    nodes = ev.get("nodes", {}) or ev.get("findings", {}) or {}
    lines.append(f"  evidence nodes: {len(nodes)}")

    strat = state.get("strategy_log", {}) or {}
    fails = strat.get("failures", []) or strat.get("failure_memory", []) or []
    if fails:
        lines.append(f"  failure memory: {len(fails)} recorded")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    query = args.query or DEFAULT_QUERY

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _configure_env(args)

    # Lazy imports — must come AFTER _configure_env so the settings singleton
    # picks up our SF_* overrides at first import.
    provider_label = _install_provider(args)

    import asyncio

    from searchos.config.settings import settings
    from searchos.harness.session import (
        SearchSession,
        close_browser_service,
        wait_for_all_evolutions,
    )

    print("=" * 72, flush=True)
    print("SearchOS — open-domain information-seeking (deep research)", flush=True)
    print(f"  mode           : {args.mode}", flush=True)
    print(f"  search backend : {provider_label}", flush=True)
    print(f"  LLM provider   : {args.provider}"
          f"{' / ' + args.model if args.model else ''}", flush=True)
    print(f"  query          : {query}", flush=True)
    print("=" * 72, flush=True)

    async def _go():
        harness = SearchSession(
            workspace_root=settings.workspace_root,
            skill_library_path=settings.skill_library_path,
        )
        try:
            return await harness.run(query)
        finally:
            await wait_for_all_evolutions(timeout=None)
            await close_browser_service()

    result = asyncio.run(_go())

    # --- Collect artifacts ---
    workspace_path = Path(result.workspace_path)
    report_src = workspace_path / "output" / "report.md"
    state_src = workspace_path / "search_state.json"

    report_md = report_src.read_text(encoding="utf-8") if report_src.exists() else ""
    (out_dir / "report.md").write_text(report_md or "(no report produced)\n", encoding="utf-8")

    state_dict: dict = {}
    if state_src.exists():
        state_dict = json.loads(state_src.read_text(encoding="utf-8"))
        (out_dir / "search_state.json").write_text(
            json.dumps(state_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary = {
        "query": query,
        "mode": args.mode,
        "search_backend": provider_label,
        "provider": args.provider,
        "model": args.model or "(provider default)",
        "verdict": result.eval_verdict,
        "coverage_score": round(result.coverage_score, 4),
        "evidence_count": result.evidence_count,
        "total_steps": result.total_steps,
        "elapsed_s": round(result.elapsed_s, 1),
        "token_usage": result.token_usage,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Print ---
    print("\n----- SOCM state -----", flush=True)
    if state_dict:
        print(_render_socm(state_dict), flush=True)
    print("\n----- Report (report.md) -----", flush=True)
    print(report_md or "(no report produced)", flush=True)
    print("\n----- Run summary -----", flush=True)
    tokens = result.token_usage.get("total_tokens", 0)
    calls = result.token_usage.get("llm_calls", 0)
    print(f"  verdict={result.eval_verdict}  coverage={result.coverage_score:.0%}  "
          f"evidence={result.evidence_count}  steps={result.total_steps}  "
          f"time={result.elapsed_s:.1f}s  tokens={tokens:,}/{calls} calls", flush=True)
    print(f"  artifacts: {out_dir}/  (report.md, search_state.json, summary.json)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
