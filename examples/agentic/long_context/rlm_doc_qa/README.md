# rlm_doc_qa — Long Document QA with Recursive Language Models

Answer questions about documents that are too large to fit in a single prompt.
The LLM uses a Python REPL to chunk the document, delegate analysis to sub-LLMs,
and synthesize a final answer — no context window limit, no GPU required.

Works with any plain-text document: markdown, source code, reports, books, logs.

## Quick Start

```bash
cd examples/agentic/long_context/rlm_doc_qa

# Build
bash build.sh

# Run — mount your document and ask a question
export ANTHROPIC_API_KEY=sk-ant-...
DOCUMENT_PATH=/path/to/your/document.txt \
QUERY="What are the main conclusions?" \
bash predict.sh
```

### OpenAI backend

```bash
export OPENAI_API_KEY=sk-...
DOCUMENT_PATH=/path/to/doc.txt \
QUERY="Summarize the methodology section" \
LLM_BACKEND=openai MODEL=gpt-4o-mini \
bash predict.sh
```

## How It Works

```
Root LLM (haiku / gpt-4o-mini)
  1. peek() → understand document structure
  2. chunk_context(size) → split into manageable pieces
  3. For each chunk:
       llm_query(chunk + question) → sub-LLM extracts relevant info
  4. llm_query(all findings + question) → synthesize final answer
  5. FINAL_VAR("final_answer") → done
```

The root LLM writes all the Python — you provide the document and question.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCUMENT_PATH` | — | **Required.** Absolute path to your document |
| `QUERY` | — | **Required.** Question to answer |
| `LLM_BACKEND` | `anthropic` | `anthropic` or `openai` |
| `MODEL` | `claude-haiku-4-5-20251001` | Root LLM model |
| `MAX_ITERATIONS` | `15` | Maximum RLM loop iterations |
| `CHUNK_SIZE` | `150000` | Characters per sub-LLM chunk |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic backend |
| `OPENAI_API_KEY` | — | Required for OpenAI backend |

## REPL Environment

The LLM has access to these variables and functions inside `repl` code blocks:

| Name | Description |
|------|-------------|
| `context` | Full document text |
| `doc_len` | Total character count |
| `query` | The user's question |
| `peek(start, end)` | Return a character slice of context |
| `chunk_context(size)` | List of `(start, end)` index tuples |
| `llm_query(prompt)` | Call a sub-LLM (~150K char capacity) |

## File Layout

```
rlm_doc_qa/
  predict.py      # RLM loop + document loading + REPL setup
  rlm_repl.py     # exec-based REPL with injected helpers
  llm_client.py   # Anthropic / OpenAI client
  Dockerfile      # python:3.11-slim + anthropic + openai
  build.sh        # docker build
  predict.sh      # docker run — mounts document directory
  example.yaml    # CVlization metadata
```

## Comparison with rlm_needle

| | `rlm_needle` | `rlm_doc_qa` |
|--|--|--|
| Input | Synthetic random text | Your own document |
| Task | Find a hidden value | Answer any question |
| Strategy | Regex search | Chunk + sub-LLM synthesis |
| REPL extras | — | `doc_len`, `query`, `chunk_context()` |
| Iterations | 1–2 typical | 3–6 typical |

## References

- [Recursive Language Models paper](https://arxiv.org/abs/2512.24601) — Zhang, Kraska, Khattab (2024)
- [alexzhang13/rlm](https://github.com/alexzhang13/rlm) — production library this is adapted from
- [alexzhang13/rlm-minimal](https://github.com/alexzhang13/rlm-minimal) — minimal reference implementation
