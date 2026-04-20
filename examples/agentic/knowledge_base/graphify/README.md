# graphify — Knowledge Base via Headless Coding Agent

> **Give any codebase a persistent memory. Ask questions in plain English.
> Get answers that cite specific functions, line numbers, and architectural
> relationships — across every file, every time, with no re-reading.**

graphify turns a directory of source files, docs, and papers into a queryable
knowledge graph that persists across sessions. The coding agent (Claude Code or
Codex) builds the graph once and queries it in seconds. On a 49-file corpus of
Andrej Karpathy's nanoGPT + minGPT + research papers, graphify reduced token
cost by **71.5×** versus reading raw files on every query.

This CVlization example runs the coding agent **headlessly** — no interactive
session, no manual steps. Drop it into a CI pipeline or call it from another
agent.

> **Attribution**: graphify is open source by [@safishamsi](https://github.com/safishamsi) ·
> [github.com/safishamsi/graphify](https://github.com/safishamsi/graphify).
> The bundled sample corpus comes from
> [`worked/example/raw/`](https://github.com/safishamsi/graphify/tree/v3/worked/example/raw).

---

## What it produces

Running `ingest.sh` on a 7-file corpus (5 Python modules + 2 Markdown docs)
takes ~2 minutes and produces:

```
corpus/graphify-out/
├── GRAPH_REPORT.md   ← god nodes, communities, surprising connections, suggested questions
├── graph.json        ← full graph (GraphRAG-ready, reusable across queries)
└── graph.html        ← interactive vis.js visualization, open in browser
```

**Sample graph report (bundled corpus):**

```
112 nodes · 157 edges · 11 communities
81% EXTRACTED (from AST) · 19% INFERRED (by agent) · 0% AMBIGUOUS

God Nodes:
  ValidationError     11 edges  — touched by every module that can fail
  storage.py           9 edges  — everything reads and writes through here
  _ensure_storage()    7 edges  — silent dependency gating all storage ops
  load_index()         7 edges  — central data access point
  validate_document()  6 edges  — gateway called by parser and API before any write

Surprising connections:
  notes.md → storage.py  "Open Question: Storage Interface Abstraction" references storage.py
  notes.md → api.py       "API Layer Notes" references api.py directly
  notes.md → processor.py Three separate notes entries converge on extract_keywords()
                           — the most debated design decision in the codebase
```

**At scale** (Karpathy corpus: nanoGPT + minGPT + 5 papers + images):

```
49 files · ~92,000 words · 285 nodes · 340 edges · 53 communities
Token cost: 6,000 input + 3,500 output  (vs ~450,000 to read all files raw)
71.5× token reduction on repeat queries

God Nodes: Value (15 edges), Training Script (11), GPT (9), FlashAttention Algorithm (7)

Surprising connection discovered:
  nanoGPT/model.py → minGPT/model.py
  from_pretrained() calls get_default_config() across repos — INFERRED, confidence 0.8
  (two separate GPT implementations sharing an undocumented interface)
```

---

## Sample queries

### "what calls storage directly?"

```
From api.py (4 calls):
  handle_get()    → load_record()     api.py:30
  handle_delete() → delete_record()   api.py:37
  handle_list()   → list_records()    api.py:45
  handle_search() → load_index()      api.py:54

From processor.py (3 calls):
  find_cross_references() → load_index()      processor.py:46
  process_and_save()      → save_processed()  processor.py:59
  reprocess_all()         → load_index()      processor.py:66

From parser.py (1 call):
  parse_and_save() → save_parsed()   parser.py:66

No intermediary/repository abstraction — all three layers call storage
directly. API is heaviest (4), processor next (3), parser lightest (1).
```

### "shortest path between parser and processor"

```
No direct edge. Three 2-hop paths, all EXTRACTED (confidence 1.0):

  parser ←[imports_from]— api       —[imports_from]→ processor
  parser ←[imports_from]— storage   —[imports_from]→ processor
  parser ←[imports_from]— validator —[imports_from]→ processor

Interpretation: parser and processor are separate pipeline stages with no
direct dependency — connected only through orchestration (api), shared
infrastructure (storage), and the intermediate validation step (validator).
```

---

## How it works

The coding agent is invoked **headlessly** inside Docker, running as the host
user. Only the credentials file is mounted from the host
(`~/.claude/.credentials.json`, writable for OAuth token refresh) — the
container's CLAUDE.md, settings, hooks, and memory are kept clean. A fail-fast
auth check runs before the full pipeline.

graphify is a coding agent skill. When `ingest.sh` triggers `/graphify .` inside
the corpus directory, the agent:

1. Detects file types (code vs. docs vs. papers vs. images)
2. Extracts code structure via tree-sitter AST — deterministic, zero LLM cost
3. Extracts semantic relationships from Markdown via parallel subagents
4. Builds a NetworkX graph and runs Leiden community detection
5. Writes `GRAPH_REPORT.md`, `graph.json`, and `graph.html` to `graphify-out/`

`query.sh` loads the persisted `graph.json` and answers natural-language
questions in seconds — no re-extraction, no re-reading source files.

---

## Layout

```
examples/agentic/knowledge_base/graphify/
├── Dockerfile          # node:20-slim + claude-code + codex + graphifyy
├── build.sh
├── ingest.sh           # build the knowledge graph  (alias: train)
├── query.sh            # query the knowledge graph  (alias: predict)
├── requirements.txt    # for no-docker mode
├── example.yaml
└── corpus/             # bundled sample (5 Python modules + 2 Markdown docs)
    ├── api.py
    ├── parser.py
    ├── validator.py
    ├── processor.py
    ├── storage.py
    ├── architecture.md
    └── notes.md
```

Outputs land in `corpus/graphify-out/` (created at runtime, git-ignored).

---

## Quickstart

### With Docker (default)

Credentials are picked up automatically by mounting
`~/.claude/.credentials.json` from your host — no API key needed if you are
already logged in to Claude Code.

```bash
./build.sh

./ingest.sh                                          # build the graph (~2 min)
./query.sh "what calls storage directly?"
./query.sh "shortest path between parser and processor"
./query.sh "which module has the most connections?"
./query.sh "what design decisions does the architecture doc mention?"
```

Pass an API key explicitly if not logged in:

```bash
ANTHROPIC_API_KEY=sk-ant-... ./ingest.sh
```

### Without Docker

```bash
npm install -g @anthropic-ai/claude-code
pip install -r requirements.txt
graphify install

CVL_NO_DOCKER=1 ./ingest.sh
CVL_NO_DOCKER=1 ./query.sh "what calls storage directly?"
```

### Using Codex

```bash
AGENT=codex OPENAI_API_KEY=sk-... ./ingest.sh
AGENT=codex OPENAI_API_KEY=sk-... ./query.sh "what calls storage directly?"
```

### Point at your own codebase

```bash
./ingest.sh /path/to/your/project
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AGENT` | `claude` | `claude` or `codex` |
| `ANTHROPIC_API_KEY` | — | Fallback when `~/.claude/.credentials.json` is absent |
| `OPENAI_API_KEY` | — | Fallback when `~/.codex/auth.json` is absent |
| `CVL_NO_DOCKER` | — | Set to `1` to skip Docker |
| `CVL_IMAGE` | `cvl-knowledge-base-graphify` | Docker image name override |
