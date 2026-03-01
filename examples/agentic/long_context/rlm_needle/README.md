# rlm_needle — Needle-in-Haystack with Recursive Language Models

Demonstrates the **Recursive Language Model (RLM)** inference paradigm on a
needle-in-haystack task. A magic number is hidden in 100K+ lines of random
text. The LLM finds it by writing Python code in a REPL loop — the context
never appears in the prompt window.

No GPU required. Uses the Anthropic or OpenAI API.

## How It Works

Standard LLMs cannot process 100K lines at once (too many tokens). RLMs solve
this by giving the LLM a Python REPL as an external memory:

```
Root LLM (claude-haiku / gpt-4o-mini)
  ├─ Writes Python in ```repl``` blocks
  ├─ REPL executes code; context lives in a variable, not in the prompt
  ├─ LLM can call llm_query() to delegate analysis to a sub-LLM
  └─ Signals done with FINAL("answer") or FINAL_VAR("var_name")
```

A typical run uses 1–2 iterations:

1. LLM writes `re.search(r'magic number is (\d+)', context)` → finds the answer
2. LLM calls `FINAL_VAR("answer")` → done

## Quick Start

```bash
cd examples/agentic/long_context/rlm_needle

# Build
bash build.sh

# Run with Anthropic (default)
export ANTHROPIC_API_KEY=sk-ant-...
bash predict.sh

# Run with OpenAI
export OPENAI_API_KEY=sk-...
LLM_BACKEND=openai MODEL=gpt-4o-mini bash predict.sh
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `anthropic` | `anthropic` or `openai` |
| `MODEL` | `claude-haiku-4-5-20251001` | Model string for chosen backend |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic backend |
| `OPENAI_API_KEY` | — | Required for OpenAI backend |
| `CONTEXT_LINES` | `100000` | Lines of synthetic text to generate |
| `MAX_ITERATIONS` | `10` | Maximum RLM loop iterations |

Larger contexts (1M lines) work the same way — the LLM's regex search is O(n)
in context size, not in token count.

## File Layout

```
rlm_needle/
  predict.py      # RLM loop: context generation, iteration, answer extraction
  rlm_repl.py     # REPL: exec-based Python sandbox with injected helpers
  llm_client.py   # Thin client for Anthropic and OpenAI APIs
  Dockerfile      # python:3.11-slim + anthropic + openai
  build.sh        # docker build
  predict.sh      # docker run with env vars
  example.yaml    # CVlization metadata
```

## Connection to Claude Code

The [brainqub3/claude_code_RLM](https://github.com/brainqub3/claude_code_RLM)
repo maps this same pattern onto Claude Code primitives:

| This example | Claude Code equivalent |
|---|---|
| Root LLM loop (`run_rlm`) | Main Claude Code conversation |
| `repl.execute(code)` | Bash tool running `rlm_repl.py exec` |
| `llm_query(prompt)` | Spawning a subagent (`/rlm-subcall`) |
| Docker container | Claude Code session sandbox |

The Docker approach here is self-contained and reproducible; the Claude Code
approach integrates naturally with existing workflows and the IDE.

## References

- [Recursive Language Models paper](https://arxiv.org/abs/2512.24601) — Zhang, Kraska, Khattab (2024)
- [alexzhang13/rlm-minimal](https://github.com/alexzhang13/rlm-minimal) — minimal reference implementation this is adapted from
- [alexzhang13/rlm](https://github.com/alexzhang13/rlm) — production library with Docker/Modal sandboxes, cost tracking, and compaction
- [brainqub3/claude_code_RLM](https://github.com/brainqub3/claude_code_RLM) — Claude Code native RLM scaffold
- [Prime Intellect blog: RLMs, the paradigm of 2026](https://www.primeintellect.ai/blog/rlm)
