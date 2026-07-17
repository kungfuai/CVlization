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

Artifacts land in `./artifacts/` (`report.md`, `search_state.json`, `summary.json`).

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

The demo disables the Explore scout's summary→evidence *replay*
(`SF_ENABLE_EXPLORE_REPLAY=false`): the scout still discovers entities and hub
pages, but every filled cell stays backed by an actual page-open (anchored
evidence with a real source URL) instead of an unsourced scout guess.

## Output — what a good run looks like

The printed **SOCM state** shows the coverage map filling in cell by cell, and
`report.md` is a table where every cell cites the corpus page it came from:

<!-- VERIFIED_OUTPUT -->

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
