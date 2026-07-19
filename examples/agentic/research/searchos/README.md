# SearchOS — multi-agent open-domain information-seeking (deep research)

Wraps **[SearchOS](https://github.com/antins-labs/SearchOS)** (Ant Group,
[arXiv:2607.15257](https://arxiv.org/abs/2607.15257)) as a thin, reproducible
CVlization example. SearchOS is a multi-agent collaboration system that answers
open-domain "deep research" questions by treating **search state as a system
asset** rather than letting it drown in conversation history.

## Why this example

CVlization already has `agentic/web/browser_use` (a *single* browser agent) and
`agentic/rag` (retrieval-augmented QA over a fixed index). Neither covers
**multi-agent open-domain information seeking** — decompose a question, scout the
space, dispatch many sub-agents in parallel, and assemble a citation-grounded
answer. SearchOS fills that gap with four ideas worth studying:

- **SOCM (Search-Oriented Context Management)** — one persistent, shared search
  state (`search_state.json`): a task queue (**Frontier**), an **Evidence Graph**
  ((entity, attribute, value, source) facts), a **Coverage Map** (normalized
  entity×attribute tables), and **Strategy/Failure Memory**. Snapshot, restore,
  replay — sub-agents run on a compact SOCM view, not the full transcript.
- **Coverage-map-driven, recall-first dispatch** — the question becomes normalized
  tables; dispatch keeps targeting *empty cells* until every one holds a sourced
  value.
- **Pipelined-parallel sub-agents** — search → open → find stages overlap across
  agents; wall-clock approaches the slowest single chain, not a serial sum.
- **Citation-grounded relational schema completion** — every cell in the answer
  carries a source; the answer traces back to the pages it came from.

## What this demo does

It runs **one** SearchOS session end-to-end and collects the artifacts:

| Artifact | What it is |
|----------|------------|
| `report.md` | the synthesized, citation-grounded answer (a table, one source per cell) |
| `search_state.json` | the full SOCM state — coverage map + evidence graph + frontier queue + failure memory |
| `summary.json` | verdict, coverage %, evidence count, steps, tokens |

### Two modes, one identical pipeline

- **`--mode offline` (default, keyless search).** SearchOS binds its web search
  onto a small pluggable `SearchProvider`. This example ships
  `local_corpus_provider.py`, which serves results from a bundled synthetic corpus
  (`corpus.json`), and sets the page-fetch backend to `search_engine` so page
  `open()` is served straight from that corpus — **no network, no search key**.
  The entire multi-agent pipeline (Explore → coverage-map schema → pipelined
  sub-agents → judge-based extraction → synthesis) runs against a fixed,
  inspectable corpus, so the run is reproducible and the grounding is checkable.
  You only need an **LLM key**.

- **`--mode web` (real web search).** The upstream SearchOS path: live **Serper**
  or **Tavily** web search + page fetch. Needs `SERPER_API_KEY` or
  `TAVILY_API_KEY` (both have free tiers) in addition to the LLM key.

The offline corpus is deliberately synthetic (fictional labs) so a faithful run
*must* find and cite the corpus pages rather than lean on the model's world
knowledge — which makes it a clean grounding test.

## Quick start

```bash
# 1) Build (CPU-only; no torch/CUDA)
bash build.sh

# 2) Offline demo — needs only an LLM key (default provider: openai)
export OPENAI_API_KEY=sk-...
bash run.sh                                   # default demo query

# custom query, still offline:
bash run.sh -- "Compare the context windows of the models in the demo corpus."

# 3) Live web search (needs a search key too)
export TAVILY_API_KEY=tvly-...                # or SERPER_API_KEY=...
bash run.sh -- --mode web "Top-5 universities per subject in the 2025 QS rankings, with deadlines"
```

Artifacts land in `searchos_outputs/` in your current directory (override with
`SEARCHOS_OUTPUTS=/path`): `report.md`, `search_state.json`, `summary.json`.

### What to expect

- **No model downloads** — SearchOS is pure API-based orchestration; the image
  is self-contained (~600 MB, built once) and runtime needs only the LLM API.
- **One run of the default demo**: ~70–110 s wall-clock, ~180k–370k tokens
  (≈ $0.5–1.0 with `gpt-4.1`/`gpt-4.1-mini`), ~30–80 LLM calls.
- **Console output**: the orchestrator's progress, then a rendered SOCM
  coverage map (cells filling in), the citation-grounded report table, and a
  one-line run summary (verdict / coverage / evidence / tokens).
- **Stochastic by nature**: the LLM designs the table schema and the dispatch
  plan, so column names and coverage differ run to run (typically 80–100%
  of cells filled within the demo budget). Unfilled cells print as `—` —
  with this example's grounding setup a cell is either backed by evidence or
  honestly missing, never guessed.

## Choosing the LLM

SearchOS uses a one-knob provider preset. Pass `--provider` and the matching key:

| `--provider` | Protocol | Key env | Notes |
|--------------|----------|---------|-------|
| `openai` (default) | OpenAI | `OPENAI_API_KEY` | verified with `--model gpt-4.1 --fast-model gpt-4.1-mini` |
| `anthropic` | Anthropic | `ANTHROPIC_API_KEY` | Console API key only |
| `deepseek` | Anthropic-protocol | `DEEPSEEK_API_KEY` | |
| `zhipu`, `zai`, `kimi-coding`, `qwen-coding`, ... | vendor coding plans | vendor key | see upstream `config/providers.py` |

`--model` / `--fast-model` override the model ids within the chosen provider (the
"fast" tier does extraction/synthesis). The default OpenAI preset models
(`gpt-5.5` / `gpt-5.4-mini`) may not be available on every account, so this
example verifies with `gpt-4.1` / `gpt-4.1-mini`.

## Options (`run_search.py`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--mode` | `offline` | `offline` (local corpus) or `web` (Serper/Tavily) |
| `--corpus` | `corpus.json` | offline corpus path |
| `--provider` | `openai` | SearchOS `SF_PROVIDER` preset |
| `--model` / `--fast-model` | provider default | override model ids |
| `--search-provider` | auto | web mode: `serper` or `tavily` |
| `--max-parallel-agents` | `3` | cap on concurrent sub-agents |
| `--max-iterations` | `24` | orchestrator loop cap |
| `--max-time-s` | `600` | wall-clock budget (0 = unlimited) |
| `--enable-skills` | off | enable the hierarchical skill system |

## Grounding — how this example keeps the answer honest

Because the offline corpus is fictional, any cell value that does not appear in
`corpus.json` is by construction a hallucination — which made three real
evidence-laundering paths easy to catch and fix while building this example
(all reproduced with `gpt-4.1` sub-agents against upstream commit `361373b`):

1. **Sub-agents answering without ever searching.** The stock search-agent
   persona opens with *"End by emitting a … message with no tool call"*;
   gpt-4.1-class models read that as the immediate instruction and answer in a
   single turn from prior knowledge — zero tool calls plus a fabricated
   citation trail (reproduced 6/6). `patch_search_persona.py` rewords the
   persona at image build time to make tool-first sequencing explicit
   (6/6 tool-using with the reword). The exact-match patch fails the build if
   the upstream persona changes.
2. **Final-summary replay.** Upstream unconditionally replays every
   sub-agent's final message into the evidence graph as
   `agent://<id>/final_summary`, so a hallucinated summary silently fills
   coverage cells. The build patch adds an `SF_ENABLE_SUMMARY_REPLAY` gate
   (default `true` = stock behavior); this example sets it to `false`, and
   likewise disables the analogous Explore-scout replay
   (`SF_ENABLE_EXPLORE_REPLAY=false`, an upstream setting). The scout still
   discovers entities and hub pages — its prose is just not evidence.
3. **Non-http source URLs are silently dropped.** SearchOS's evidence
   extraction only attributes sources matching `https?://`; the corpus
   originally used a `corpus://` scheme and every page-anchored citation lost
   its URL. The corpus now uses reserved `.example`-TLD https URLs (RFC 2606;
   never fetched — offline mode serves every page from the local corpus).

Remaining provenance tiers you will see in `search_state.json`:
**page-anchored** evidence (a real corpus URL plus a verbatim
`source_excerpt` from that page — the bulk of filled cells),
**orchestrator-asserted** (`agent://orchestrator/asserted`, an upstream
orchestrator tool for relaying a value verbatim from a sub-agent's summary;
lower confidence, clearly labeled), and **explore-seeded** identity cells
(the entity names themselves). Audit any run with two greps: every value
should appear in `corpus.json`, and every anchored `source_excerpt` should be
a verbatim quote of its source page.

## Output — what a good run looks like

The printed **SOCM state** shows the coverage map filling in cell by cell, and
`report.md` is a table where every cell cites the corpus page it came from.
Verified run (2026-07-17, `--model gpt-4.1 --fast-model gpt-4.1-mini`,
offline mode; 16/16 cells filled, every value matches `corpus.json`, 9/12
attribute cells page-anchored with verbatim excerpts):

The complete artifact bundle from that run is committed in
[`sample_run/`](sample_run/) —
[`report.md`](sample_run/report.md) (the citation-grounded answer),
[`summary.json`](sample_run/summary.json) (verdict / coverage / tokens),
and [`search_state.json`](sample_run/search_state.json) (**the central
artifact**: the full SOCM state — coverage map, evidence graph with per-cell
source URLs and verbatim `source_excerpt` quotes, frontier queue, strategy
log). The excerpts below are a preview of those files:

```
  Table `AI Research Labs`  (4 rows × 4 cols)
    row                     lab_name        founding_year   headquarters_c  flagship_model
    Nimbus Labs               ✓               ✓               ✓               ✓
    Orchard AI                ✓               ✓               ✓               ✓
    Basilisk Research         ✓               ✓               ✓               ✓
    Tidewater Institute       ✓               ✓               ✓               ✓
  cells filled: 16/16
  evidence nodes: 16
```

```markdown
| lab_name | founding_year | headquarters_city | flagship_model |
|---|---|---|---|
| Basilisk Research | 2018 [2] | Tallinn [2] | Gaze-XL [3] |
| Nimbus Labs | 2019 [4] | Reykjavik [5] | StormCast-3 [6] |
| Orchard AI | 2021 [7] | Lisbon [7] | Pippin-1B [8] |
| Tidewater Institute | 2016 [9] | Hobart [9] | Ebb-Sim [9] |

[2] https://basilisk-research.example/home
[3] https://basilisk-research.example/gaze
[4] https://nimbus-labs.example/about
[5] https://nimbus-labs.example/contact
[6] https://nimbus-labs.example/models
[7] https://orchard-ai.example/about
[8] https://orchard-ai.example/flagship
[9] agent://orchestrator/asserted
```

```
verdict=COMPLETE  coverage=100%  evidence=16  steps=32  time=69.4s  tokens=220,369/36 calls
```

## Notes & caveats

- **CPU-only.** SearchOS is pure API-based LLM orchestration (langchain /
  langgraph / deepagents) — no torch, no CUDA.
- **Cost/latency scale with model + question.** The demo is bounded
  (`--max-iterations`, `--max-parallel-agents`, `--max-time-s`) and uses a small
  corpus so it converges quickly.
- **Grounding depends on the model.** Smaller/weaker models may still occasionally
  mis-extract; the offline corpus makes such errors easy to spot against
  `corpus.json`.
- SearchOS is pinned to a specific upstream commit in the `Dockerfile` for
  reproducibility.

## References

- SearchOS repo: https://github.com/antins-labs/SearchOS
- Paper: https://arxiv.org/abs/2607.15257 · https://huggingface.co/papers/2607.15257
