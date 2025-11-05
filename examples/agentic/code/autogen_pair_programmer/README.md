# AutoGen Pair Programmer

Pair-programming example that demonstrates how Microsoft AutoGen can coordinate multiple agents (Coder, Reviewer, Manager) to solve a simple coding task. The project ships with a deterministic fallback so you can explore the workflow without a real LLM; if you supply OpenAI credentials, the script will attempt an actual AutoGen conversation and then run the unit tests.

## Directory Overview

```
examples/agentic/code/autogen_pair_programmer/
├── Dockerfile
├── requirements.txt
├── build.sh
├── run.sh            # Run a pair-programming session
├── evaluate.sh       # Smoke test / CI helper
├── pair_programmer.py
├── evaluate.py
└── tasks/simple_math/
    ├── task.py       # Function to implement
    └── tests/        # Pytest suite
```

## Quick Start

```bash
# Build the container
bash build.sh

# Run with the mock provider (no external LLM)
bash run.sh --task simple_math

# Optional: switch to OpenAI via LiteLLM-compatible config
export PAIR_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
bash run.sh --llm-model openai/gpt-4o-mini --task simple_math

# Evaluate (uses mock fallback so it always passes offline)
bash evaluate.sh
```

## Providers

| Provider | How to enable | Notes |
|----------|---------------|-------|
| `mock` (default) | leave flags unset | Deterministic patch ensures tests pass; conversation is simulated. |
| `openai` | set `PAIR_LLM_PROVIDER=openai` and `OPENAI_API_KEY` | The script attempts an AutoGen conversation before applying the fallback. |

Other providers (Groq, Ollama, Azure) are not wired up yet; the script will fall back to the deterministic path and note that in the logs.

## Workflow

1. The script backs up the task file.
2. If a supported LLM is configured, it launches a Manager–Coder–Reviewer trio using AutoGen.
3. Regardless of the LLM result, a reference implementation is applied to ensure the tests succeed.
4. Pytest validates the task and the backup is restored, making the example idempotent.

Conversation logs are written to `var/logs/session_<task>.json` for inspection.

## Evaluation

`evaluate.sh` invokes `pair_programmer.py --json` in mock mode and verifies the Python tests. The script exits non-zero if anything fails, making it CI friendly.
