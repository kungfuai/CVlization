# LangGraph Helpdesk RAG Agent

Retrieval-augmented helpdesk agent built with LangChain + LangGraph. The agent indexes bundled CVlization support docs into a Chroma vector store, plans retrieval via LangGraph, and answers user questions with citations.

## Key Features

- **Self-contained RAG stack**: HuggingFace embeddings + Chroma vector store persisted under `var/chroma`.
- **LangGraph workflow**: Retrieval node feeds an LLM node that formats answers with `[doc:n]` citations (or a deterministic mock responder for offline testing).
- **Pluggable LLMs**: Works with Ollama, OpenAI, or Groq via environment variables; defaults to a `mock` responder so the example works out-of-the-box.
- **Evaluation harness**: Lightweight heuristic scoring (answer substring + context coverage) on curated eval prompts.

## Directory Layout

```
examples/agentic/rag/langgraph_helpdesk
├── Dockerfile
├── build.sh          # Build Docker image
├── ingest.sh         # Ingest support docs into Chroma
├── predict.sh        # Answer a user question
├── query.sh          # Alias for predict.sh
├── evaluate.sh       # Heuristic evaluation on sample QA set
├── ingest.py / predict.py / evaluate.py / pipeline.py
├── data/
│   ├── docs/         # Bundled helpdesk markdown files
│   └── evals/        # JSONL eval questions & answers
└── example.yaml
```

All scripts mount the repository root at `/cvlization_repo` and reuse `~/.cache/cvlization` for model downloads.

## Quick Start

```bash
# 1) Build image
bash build.sh

# 2) Create vector store (add --reset to rebuild)
bash ingest.sh

# 3) Ask a question (mock LLM by default)
bash query.sh --question "How do I share model downloads across examples?"
# Use a hosted model (example: OpenAI)
# bash query.sh --llm-provider openai --llm-model gpt-4o-mini --question "How do I share model downloads across examples?"

# 4) Run heuristic evaluation
bash evaluate.sh
```

### Sample Output

```
Q: How do I share model downloads across examples?
A: - All examples share `~/.cache/cvlization` for models. [doc:1]
```

## LLM Configuration

| Provider | How to enable | Notes |
|----------|---------------|-------|
| Mock (default) | leave flags/env unset | Deterministic response pulled from top context chunk (no external LLM required) |
| Ollama | `--llm-provider ollama` or `export HELPDESK_LLM_PROVIDER=ollama` | Requires `ollama serve`; override model via `--llm-model` / `HELPDESK_LLM_MODEL` (default `mistral`) |
| OpenAI | `--llm-provider openai` + `OPENAI_API_KEY` | Override with `--llm-model` / `HELPDESK_LLM_MODEL` (default `gpt-4o-mini`) |
| Groq | `--llm-provider groq` + `GROQ_API_KEY` | Override with `--llm-model` / `HELPDESK_LLM_MODEL` (default `llama3-8b-8192`) |

CLI flags (`--llm-provider`, `--llm-model`, `--temperature`) take precedence. Environment fallbacks are `HELPDESK_LLM_PROVIDER`, `HELPDESK_LLM_MODEL`, and `HELPDESK_LLM_TEMPERATURE` (legacy names `HELPDESK_LLM`, `HELPDESK_MODEL`, `HELPDESK_TEMPERATURE` still work).

## Re-ingesting Documents

Place additional `.md`/`.txt` files in `data/docs/` and rerun:

```bash
bash ingest.sh --reset
```

The script rebuilds the Chroma store using `sentence-transformers/all-MiniLM-L6-v2`.

### Optional: Import AskUbuntu Dataset

To populate a richer corpus without leaving Docker, fetch a subset of the public
[`maifeeulasad/askubuntu-data`](https://huggingface.co/datasets/maifeeulasad/askubuntu-data)
dataset and convert it into markdown files:

```bash
# Make sure the image exists
bash build.sh

# Download the first 10k posts into data/docs/askubuntu/
docker run --rm \
  --workdir /workspace \
  --mount "type=bind,src=$(pwd),dst=/workspace" \
  langgraph_helpdesk \
  python fetch_askubuntu_subset.py --limit 10000
```

This writes `askubuntu_*.md` files under `data/docs/askubuntu/`. Run
`bash ingest.sh --reset` afterward to rebuild the vector store. Adjust `--limit`
to control corpus size (set `--limit 0` to stream the full dataset). Note the dataset
is licensed under AGPL-3.0—ensure it fits your usage.

## Evaluation

`evaluate.sh` loads `data/evals/helpdesk_eval.jsonl`, runs each question through the agent, and reports:

- Answer hit rate (does the model output contain the ground truth string?)
- Context coverage rate (does any retrieved context contain the ground truth string?)

It prints a brief summary plus a few sample outputs; extend the JSONL file to cover more scenarios.

## Metadata

All metadata for discovery lives in `example.yaml`. Update tags or resource estimates there when the example evolves.
