# graphify — Knowledge Base Example

Build and query a persistent knowledge graph from a code corpus using a headless coding agent (Claude Code or Codex).

graphify transforms a directory of source files into a navigable knowledge graph: call graphs, import relationships, and cross-file semantic connections. The graph persists across sessions in `graphify-out/` and reduces token cost on repeat queries.

This CVlization example invokes the coding agent headlessly — no interactive session needed.

## Layout

```
examples/agentic/knowledge_base/graphify/
├── Dockerfile          # node:20-slim + claude-code + codex + graphifyy
├── build.sh
├── ingest.sh           # build the knowledge graph (alias: train)
├── query.sh            # query the knowledge graph (alias: predict)
├── requirements.txt    # for no-docker mode
├── example.yaml
└── corpus/             # bundled sample (5 Python files + 2 Markdown docs)
    ├── api.py
    ├── parser.py
    ├── validator.py
    ├── processor.py
    ├── storage.py
    ├── architecture.md
    └── notes.md
```

## Quickstart

### With Docker (default)

Credentials are picked up automatically by mounting `~/.claude` (or `~/.codex`) from your host. No API key env var needed if you are already logged in.

```bash
# Build the image
./build.sh

# Build the knowledge graph (mounts ~/.claude automatically)
./ingest.sh

# Query the graph
./query.sh "what calls storage directly?"
./query.sh "shortest path between parser and processor"
./query.sh "which module has the most connections?"
```

If you are not logged in interactively, pass the API key explicitly:

```bash
ANTHROPIC_API_KEY=sk-ant-... ./ingest.sh
```

### Without Docker

```bash
# Install dependencies
npm install -g @anthropic-ai/claude-code   # or: npm install -g @openai/codex
pip install -r requirements.txt
graphify install                            # or: graphify install --platform codex

# Build the knowledge graph
CVL_NO_DOCKER=1 ANTHROPIC_API_KEY=sk-ant-... ./ingest.sh

# Query the graph
CVL_NO_DOCKER=1 ANTHROPIC_API_KEY=sk-ant-... ./query.sh "what calls storage directly?"
```

### Using Codex instead of Claude Code

```bash
AGENT=codex OPENAI_API_KEY=sk-... ./ingest.sh
AGENT=codex OPENAI_API_KEY=sk-... ./query.sh "what calls storage directly?"
```

### Point at your own codebase

```bash
./ingest.sh /path/to/your/project
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AGENT` | `claude` | Coding agent to use: `claude` or `codex` |
| `ANTHROPIC_API_KEY` | — | Fallback when `AGENT=claude` and `~/.claude/.credentials.json` is absent |
| `OPENAI_API_KEY` | — | Fallback when `AGENT=codex` and `~/.codex/auth.json` is absent |
| `CVL_NO_DOCKER` | — | Set to `1` to skip Docker |
| `CVL_IMAGE` | `cvl-knowledge-base-graphify` | Docker image name override |

## What the corpus demonstrates

The bundled corpus is a small document pipeline (parser → validator → processor → storage → api) with architecture notes. After `ingest.sh`:

- `api.py` appears as a hub node connected to all four modules
- `storage.py` is the highest-degree god node (everything reads/writes through it)
- `architecture.md` and `notes.md` are linked to the code modules they discuss
- Two communities: the Python modules cluster together, the markdown docs cluster separately

Outputs land in `graphify-out/`:
- `GRAPH_REPORT.md` — god nodes, communities, surprising connections
- `graph.json` — full graph (GraphRAG-ready)
- `graph.html` — interactive vis.js visualization

## How it works

graphify is a coding agent skill. When `ingest.sh` runs `/graphify ./corpus`, the agent:
1. Detects file types (code vs. docs)
2. Extracts code structure via tree-sitter AST (no LLM tokens)
3. Extracts semantic relationships from markdown via subagents
4. Builds a NetworkX graph, runs Leiden community detection
5. Writes `GRAPH_REPORT.md`, `graph.json`, and `graph.html`

`query.sh` then feeds the graph report and JSON back to the agent to answer natural-language questions via graph traversal.
