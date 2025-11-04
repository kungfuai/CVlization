# Smolagents Data Analyst

Smolagents-powered tool-use example that transforms a marketing KPI dataset into insights. The agent can query DuckDB, run custom Python snippets, and (optionally) call a hosted LLM via LiteLLM. When no LLM credentials are supplied, the example falls back to a deterministic rule-based summary so you can explore the workflow offline.

## Directory Layout

```
examples/agentic/tool_use/smolagents_data_analyst
├── Dockerfile
├── requirements.txt
├── build.sh
├── predict.sh
├── evaluate.sh
├── predict.py
├── evaluate.py
└── data/marketing_kpi.csv
```

## Quick Start

```bash
# 1. Build the Docker image
bash build.sh

# 2. Run a question (mock mode)
bash predict.sh --question "Which segment generates the most revenue?"

# 3. (Optional) Switch to OpenAI via LiteLLM
export ANALYST_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
bash predict.sh --llm-model openai/gpt-4o-mini --question "What KPIs look strongest?"

# 4. Evaluate the pipeline
bash evaluate.sh
```

## LLM Providers

| Provider (flag/env) | Notes |
|---------------------|-------|
| `mock` (default) | Uses the deterministic summary so the example works offline. |
| `openai` | Set `OPENAI_API_KEY`; override model with `--llm-model` (defaults to `openai/gpt-4o-mini`). |
| `groq` | Set `GROQ_API_KEY`; defaults to `groq/llama3-8b-8192`. |
| `ollama` | Ensure Ollama is running on the host (`ollama serve`). Set `OLLAMA_BASE_URL` if not using `http://host.docker.internal:11434`. |

Command-line flags `--llm-provider`, `--llm-model`, and `--temperature` override environment variables (`ANALYST_LLM_PROVIDER`, `ANALYST_LLM_MODEL`, `ANALYST_LLM_TEMPERATURE`).

## Tools available to the agent

- `duckdb_sql`: Executes SQL against the in-memory `marketing_kpi` table.
- `python_interpreter`: Runs small Python snippets with access to pandas/duckdb.

## Evaluation

`evaluate.sh` runs three benchmark questions and checks that the answer mentions the correct customer segment, region, and headline revenue figures. The script exits with code 1 if any scenario fails, making it suitable for CI.

## Notes

- The example requires `litellm` to route provider APIs. Additional keys (e.g. `LITELLM_API_KEY`) can also be passed through the container if needed.
- For Ollama integration inside Docker, the default base URL is `http://host.docker.internal:11434`. Adjust via `OLLAMA_BASE_URL` if your setup differs.
