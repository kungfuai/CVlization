# rlm_claude_code — Long Context QA with Claude Code in Docker

Runs **Claude Code CLI** inside a Docker container to answer questions about
large documents. The example directory is mounted as `/workspace`, so Claude
Code automatically discovers the RLM skill and subagent from `.claude/`.

Claude orchestrates the analysis natively — it chooses the right strategy
(grep, chunked reads, sub-LLM delegation) without a custom RLM loop.
No GPU required. Requires only `ANTHROPIC_API_KEY`.

## Quick Start

```bash
cd examples/agentic/long_context/rlm_claude_code

# Build
bash build.sh

# Run
export ANTHROPIC_API_KEY=sk-ant-...
DOCUMENT_PATH=/path/to/your/document.txt \
QUERY="What are the main conclusions?" \
bash predict.sh
```

## How It Works

```
Docker container (/workspace = this example directory)
  │
  ├── Claude Code CLI  (claude --print --dangerously-skip-permissions)
  │     │
  │     ├── /rlm skill  (.claude/skills/rlm/SKILL.md)
  │     │     ├── Bash: grep, python3 rlm_repl.py exec ...
  │     │     └── Spawns rlm-subcall subagent per chunk
  │     │
  │     └── rlm-subcall subagent  (.claude/agents/rlm-subcall.md)
  │           └── Haiku model, Read tool only, returns JSON
  │
  └── Document mounted at /docs/ (read-only)
```

### Why mount the example directory?

CVlization mounts `SCRIPT_DIR` as `/workspace` at runtime (source code is not
baked into the image). This means `.claude/skills/` and `.claude/agents/` land
exactly where Claude Code expects them — no path configuration needed.

### What Claude Code does with the `/rlm` skill

1. Initialises a persistent Python REPL (`rlm_repl.py`) with the document
2. Peeks at structure to choose chunking strategy
3. Writes chunks to `.claude/rlm_state/chunks/`
4. Spawns `rlm-subcall` (Haiku) per chunk — returns structured JSON findings
5. Synthesises across all findings for the final answer

For small documents or simple searches, Claude may use `grep` or `Read`
directly without going through the full REPL pipeline — this is correct
behaviour.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCUMENT_PATH` | — | **Required.** Path to your document |
| `QUERY` | — | **Required.** Question to answer |
| `ANTHROPIC_API_KEY` | — | **Required.** Anthropic API key |
| `MODEL` | `claude-sonnet-4-6` | Root LLM model |
| `IMG` | `rlm-claude-code` | Docker image name |

## File Layout

```
rlm_claude_code/
  .claude/
    agents/
      rlm-subcall.md          # Haiku subagent: chunk analysis → JSON
    skills/
      rlm/
        SKILL.md              # RLM skill: orchestration procedure
        scripts/
          rlm_repl.py         # Persistent Python REPL (dep-free, ~350 LOC)
    rlm_state/
      .gitignore              # Excludes runtime state from git
  Dockerfile                  # node:20-slim + claude CLI + python3
  build.sh
  predict.sh
  example.yaml
  README.md
```

## Comparison with the other long_context examples

| | `rlm_needle` | `rlm_doc_qa` | `rlm_claude_code` |
|--|--|--|--|
| LLM backend | Anthropic or OpenAI | Anthropic or OpenAI | Anthropic only |
| RLM loop | Custom Python | Custom Python | Claude Code native |
| Sub-LLM | `llm_query()` in REPL | `llm_query()` in REPL | `rlm-subcall` subagent |
| Strategy fixed | Yes (regex/chunking) | Yes (chunking+synthesis) | No — Claude decides |
| Reproducible | High | High | Medium (model-dependent) |
| Use when | Demo / CI | Pipeline / any API | Interactive / IDE use |

## References

- [Recursive Language Models paper](https://arxiv.org/abs/2512.24601) — Zhang, Kraska, Khattab (2024)
- [brainqub3/claude_code_RLM](https://github.com/brainqub3/claude_code_RLM) — original Claude Code RLM scaffold this adapts
- [alexzhang13/rlm](https://github.com/alexzhang13/rlm) — production Python RLM library
