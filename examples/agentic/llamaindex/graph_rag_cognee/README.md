# LlamaIndex Cognee GraphRAG

This example takes inspiration from the official LlamaIndex + Cognee integration demo without depending on Cognee services. It ingests a couple of short biographies, builds a graph-backed retrieval pipeline, and surfaces the graph answers, RAG answers, and related nodes. Mock mode avoids external LLMs by returning deterministic answers derived from the sample documents.

## Directory Layout

```
examples/agentic/llamaindex/graph_rag_cognee/
├── Dockerfile
├── requirements.txt
├── build.sh
├── pipeline.py          # Query helpers for Cognee GraphRAG and mock fallback
├── query.py / query.sh  # CLI interface
├── evaluate.py / evaluate.sh
├── outputs/             # Saved answers (generated)
└── .cache/lancedb/      # Created at runtime for vector storage
```

## Quick Start

```bash
# Build the Docker image
bash build.sh

# Run the mock pipeline (no external LLMs)
bash query.sh --provider mock "Tell me who are the people mentioned?"
cat outputs/result.json

# Mock-mode smoke test
bash evaluate.sh
```

## Real Providers

| Provider | Env vars | Notes |
|----------|----------|-------|
| `openai` | `OPENAI_API_KEY`, optional `LLAMA_GRAPHRAG_OPENAI_MODEL` | Mirrors the notebook defaults (GPT-4o-mini). |
| `hf` (experimental) | optional `LLAMA_GRAPHRAG_HF_MODEL`, requires a HuggingFace text-generation model | Quality depends on the local model's ability to follow instructions. |
| `mock` (default) | none | Deterministic answers lifted from the sample documents. |

Example with OpenAI:

```bash
export LLAMA_GRAPHRAG_PROVIDER=openai
export OPENAI_API_KEY=sk-...
bash query.sh "Tell me who are the people mentioned?"
```

The vector/graph stores live under `.cache/lancedb` inside the example directory so repeated runs reuse the processed data.
