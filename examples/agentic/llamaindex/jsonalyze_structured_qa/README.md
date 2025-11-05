# LlamaIndex JSONalyze Structured QA

This example adapts the LlamaIndex JSONalyze workflow into a reproducible Docker setup. It loads a list of JSON records into an in-memory SQLite database, lets an LLM compose SQL via the JSONalyze prompt, executes the query, and synthesizes an answer. A deterministic "mock" mode is also included for offline smoke tests by applying hand-written heuristics over the sample dataset.

## Directory Layout

```
examples/agentic/llamaindex/jsonalyze_structured_qa/
├── Dockerfile
├── requirements.txt
├── build.sh
├── pipeline.py          # Core query helpers (LLM + workflow + mock)
├── query.py / query.sh  # CLI entry point
├── evaluate.py / evaluate.sh
├── data/
│   └── people.json      # Sample structured dataset
└── outputs/             # Generated answers (created at runtime)
```

## Quick Start

```bash
# Build the Docker image
bash build.sh

# Run a question in mock mode (no external LLM calls)
bash query.sh --provider mock "What is the maximum age among the individuals?"
cat outputs/result.json

# Basic regression check (mock mode)
bash evaluate.sh
```

## Switching Providers

Set `LLAMA_JSONALYZE_PROVIDER` before invoking the scripts:

| Provider | Env vars | Notes |
|----------|----------|-------|
| `mock` (default) | none | Rule-based SQL templates covering the bundled examples. |
| `openai` | `OPENAI_API_KEY`, optional `LLAMA_JSONALYZE_OPENAI_MODEL` | Defaults to `gpt-4o-mini`. |
| `hf` (HuggingFace) | optional `LLAMA_JSONALYZE_HF_MODEL` | Defaults to `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. |

Example:

```bash
export LLAMA_JSONALYZE_PROVIDER=openai
export OPENAI_API_KEY=sk-...
bash query.sh "How many individuals have a major in Psychology?"
```

## Implementation Notes

- The real-provider path mirrors the official LlamaIndex workflow: JSON records are inserted into `sqlite-utils`, a text-to-SQL prompt generates a query, and the answer is synthesized via `DEFAULT_RESPONSE_SYNTHESIS_PROMPT`.
- Mock mode uses the same dataset but maps a handful of question templates to deterministic SQL snippets so that CI can run without external LLMs.
- Outputs are saved to `outputs/result.json` and include the answer string, the generated (or heuristic) SQL, and the resulting rows.
