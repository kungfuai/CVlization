# DSPy + GEPA Prompt Optimization

Demonstrates how to refine a helpdesk prompt using [DSPy](https://github.com/stanfordnlp/dspy) and the GEPA reflective optimizer. By default the example runs in `mock` mode, which skips the LLM calls and produces deterministic scores. Switching to a real provider (OpenAI, Groq, Ollama) lets GEPA iteratively evolve the instructions.

## Directory Layout

```
examples/agentic/optimization/dspy_gepa_promptops/
├── Dockerfile
├── requirements.txt
├── build.sh
├── optimize.py      # Main optimization entrypoint
├── optimize.sh      # Container wrapper
├── evaluate.py
├── evaluate.sh
└── data/
    ├── train.jsonl
    └── dev.jsonl
```

## Quick Start

```bash
bash build.sh

# Run optimization in mock mode (no external LLM)
bash optimize.sh
cat var/results.json

# Evaluate (ensures optimized score ≥ baseline)
bash evaluate.sh
```

## Switching to a Real Provider

Set the provider and credentials before invoking `optimize.sh`/`evaluate.sh`:

```bash
export DSPY_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
# optional: export DSPY_LLM_MODEL=openai/gpt-4o-mini
bash optimize.sh --max-metric-calls 40
```

Supported providers in this example:

| Provider | Environment Variables | Notes |
|----------|----------------------|-------|
| `mock` (default) | none | Deterministic fallback that appends guidance to the prompt. |
| `openai` | `OPENAI_API_KEY`, optional `DSPY_LLM_MODEL` | Uses DSPy`s OpenAI integration (default `openai/gpt-4o-mini`). |
| `groq` | `GROQ_API_KEY`, optional `DSPY_LLM_MODEL` | Defaults to `groq/llama3-8b-8192`. |
| `ollama` | `OLLAMA_BASE_URL` (optional), `DSPY_LLM_MODEL` (default `ollama/llama3.1`) | Assumes Ollama is accessible from the container. |

## What the scripts do

1. Load small train/dev splits of helpdesk Q&A pairs.
2. Configure DSPy with the chosen provider.
3. Run GEPA (or the mock fallback) to improve the instructions for a simple `ChainOfThought` module.
4. Evaluate the baseline versus optimized prompt on the dev set and emit `var/results.json`.

## Evaluation

`evaluate.sh` reruns `optimize.py` and asserts that the optimized score is at least as good as the baseline score. This keeps CI green even in mock mode.
