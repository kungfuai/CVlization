# LlamaIndex ReAct Finance Query Agent

This example ports the official LlamaIndex “ReAct Agent with Query Engine (RAG) Tools” notebook into a reproducible Docker workflow. It builds vector indexes over the 2021 Lyft and Uber 10-K filings and exposes a ReAct-style agent that chooses the appropriate query engine (or both) when answering financial questions.

## Directory Layout

```
examples/agentic/llamaindex/react_finance_query_agent/
├── Dockerfile
├── requirements.txt
├── build.sh
├── ingest.py             # Download 10-K PDFs and build vector indexes
├── ingest.sh
├── pipeline.py           # Agent factory (OpenAI, HuggingFace, or mock)
├── query.py              # CLI entry point
├── query.sh
├── evaluate.py           # Smoke test in mock mode
├── evaluate.sh
├── data/
│   └── raw/              # Populated at runtime with Lyft/Uber PDFs
├── outputs/              # Generated answers
├── scripts/
│   └── download_10k.py   # Fetch documents with checksum verification
└── storage/              # Persisted indexes (created during ingest)
```

## Quick Start

```bash
# Build the Docker image
bash build.sh

# Ingest documents and build indexes (uses HuggingFace embeddings by default)
bash ingest.sh

# Ask a question using the deterministic mock agent
bash query.sh "What was Lyft's revenue growth in 2021?"
cat outputs/response.txt

# Run the smoke test (ensures Lyft/Uber mentions appear)
bash evaluate.sh
```

The default `mock` provider avoids external API calls and simply surfaces retrieved context. For production-quality answers, switch to a real LLM provider.

## Switching Providers

Set `LLAMA_REACT_PROVIDER` before invoking the scripts:

| Provider | Env vars | Notes |
|----------|----------|-------|
| `mock` (default) | none | Deterministic output built from retrieved chunks. |
| `openai` | `OPENAI_API_KEY`, optional `LLAMA_OPENAI_MODEL` | Defaults to `gpt-4o-mini`. |
| `hf` (HuggingFace) | optional `LLAMA_HF_MODEL` | Defaults to `google/flan-t5-base` (CPU-friendly). |

Example:

```bash
export LLAMA_REACT_PROVIDER=openai
export OPENAI_API_KEY=sk-...
bash ingest.sh          # reuse existing storage if already built
bash query.sh "Compare Uber and Lyft revenue growth in 2021."
```

## Data

`scripts/download_10k.py` downloads the Lyft and Uber 2021 10-K PDFs from the LlamaIndex examples GitHub repository and validates them via SHA256 checksums. Files are stored under `data/raw/` and the resulting indexes under `storage/{lyft,uber}`.

## Evaluation

`evaluate.py` runs two queries in mock mode to ensure both companies’ contexts appear in the answer. For real-provider validation, re-run with `LLAMA_REACT_PROVIDER=openai` (or `hf`) so the agent performs full tool selection with your chosen LLM.
