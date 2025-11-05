# Offline LLaMA.cpp Finance Assistant

This example packages a fully offline retrieval-augmented assistant powered by [llama.cpp](https://github.com/ggerganov/llama.cpp). It ingests short excerpts from the 2021 Lyft and Uber 10-K filings, builds embeddings with a compact GGUF encoder, and answers questions with a Phi-2 GGUF language model—no external APIs required.

## Contents

```
examples/agentic/local/llamacpp_assistant/
├── Dockerfile
├── build.sh          # Build the llama.cpp CPU image
├── ingest.sh         # Download models (if needed) + create embedding index
├── query.sh          # Run offline RAG inference
├── evaluate.sh       # Smoke tests over bundled questions
├── pipeline.py       # Shared ingestion + retrieval + generation logic
├── scripts/download_models.py
├── data/source/*.txt
├── models/           # Cached GGUF files (downloaded at runtime)
├── storage/          # Persisted embeddings / metadata
└── outputs/          # Query + eval artefacts
```

## Models

The helper script downloads two compact GGUF models by default:

- **Phi-2.Q4_K_M.gguf** (≈1.7 GB) — instruction-tuned chat/generation model
- **nomic-embed-text-v1.5.f16.gguf** (≈150 MB) — sentence embedding encoder

Set `LLAMACPP_LLM_PATH` or `LLAMACPP_EMBED_PATH` to point at pre-downloaded GGUF files, or disable downloads entirely with `LLAMACPP_DOWNLOAD_MODELS=0`.

## Quickstart

```bash
cd examples/agentic/local/llamacpp_assistant
./build.sh
./ingest.sh            # downloads models (once) and builds vector store
./query.sh "Compare Lyft and Uber revenue growth in 2021."
./evaluate.sh          # optional smoke tests
```

`query.sh` writes the structured response to `outputs/response.json`. Pass `--json` for raw JSON output or `--top-k` to change retrieval depth.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMACPP_DOWNLOAD_MODELS` | Allow automatic GGUF downloads | `1` |
| `LLAMACPP_LLM_PATH` | Absolute path to a local GGUF chat model | *(downloaded)* |
| `LLAMACPP_EMBED_PATH` | Absolute path to a local GGUF embedding model | *(downloaded)* |
| `LLAMACPP_THREADS` | CPU threads for llama.cpp inference | host cores |
| `LLAMACPP_TOP_K` | Retrieved chunk count | `4` |
| `LLAMACPP_TEMPERATURE` | Sampling temperature for responses | `0.1` |
| `LLAMACPP_MAX_TOKENS` | Max generation tokens | `512` |

## Notes

- All computation runs on CPU inside the container; adjust `LLAMACPP_THREADS` for performance.
- Documents live under `data/source/` to keep the example self-contained. Replace or extend these files to adapt the assistant to different domains.
- Embeddings are stored as `storage/index.npz` with metadata in `storage/chunks.json`. Delete these if you modify documents and want to rebuild from scratch.
