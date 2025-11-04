# Agentic Examples Plan

This plan tracks the three agentic example directories we intend to add under `examples/agentic/`. Each entry captures the core idea, required components, and immediate next actions so we can scaffold them consistently with the rest of the repository.

## 1. RAG Helpdesk (LangGraph)
- **Path:** `examples/agentic/rag/langgraph_helpdesk`
- **Goal:** LangChain + LangGraph retrieval-augmented answering pipeline for a helpdesk knowledge base using a local vector store.
- **Key Pieces:**
  - Docker image with Python 3.11, LangChain, LangGraph, Chroma or pgvector, Ollama or llama.cpp client.
  - Scripts: `build.sh`, `predict.sh`, optional `evaluate.sh` using RAGAS.
  - Assets: sample support documents and embeddings bootstrap.
- **Next Steps:**
  1. Draft README covering setup, ingestion, inference, and evaluation.
  2. Author base Dockerfile and shell scripts following existing template.
  3. Implement ingestion + query scripts; wire evaluation harness.

## 2. Tool-Using Data Analyst (smolagents)
- **Path:** `examples/agentic/tool_use/smolagents_data_analyst`
- **Goal:** Hugging Face `smolagents` agent that alternates between natural-language instructions and Python/DuckDB tool calls to answer business questions.
- **Key Pieces:**
  - Docker dependencies: smolagents, duckdb, pandas, optional serpapi/duckduckgo search.
  - Scripts: `build.sh`, `predict.sh`, optional `train.sh` (fine-tune phi-3-mini on tool-selection traces).
  - Sample CSV datasets and tool configuration manifest.
- **Next Steps:**
  1. Write README explaining agent loop, tools, and configuration knobs.
  2. Prepare Dockerfile with deterministic versions and caching strategy.
  3. Implement main agent entrypoint plus optional fine-tuning utilities.

## 3. Pair-Programming Agents (AutoGen)
- **Path:** `examples/agentic/code/autogen_pair_programmer`
- **Goal:** Microsoft AutoGen based Architect/Coder agents collaborating against a sandboxed shell and pytest runner to iteratively solve coding tasks.
- **Key Pieces:**
  - Docker requirements: autogen, openai-compatible client (e.g., Azure/Local), uvicorn for optional web UI, pytest.
  - Scripts: `build.sh`, `predict.sh` (run a task), `test.sh` (local validation).
  - Configuration: prompt templates, allowed commands, test harness sample task.
- **Next Steps:**
  1. Draft README covering environment variables, workflow, and safety constraints.
  2. Create Dockerfile pinned to supported Python + system packages.
  3. Implement orchestration script with shell tool wrapper and logging.

## 4. Prompt Optimization (DSPy + GEPA)
- **Path:** `examples/agentic/optimization/dspy_gepa_promptops`
- **Goal:** Demonstrate how to collect agent traces, score them with a feedback-aware metric, and use DSPy’s GEPA reflective optimizer to evolve prompts/instructions for better accuracy, tool selection, and guardrails.
- **Key Pieces:**
  - Docker image with DSPy, `gepa`, LiteLLM (or local OpenAI-compatible endpoint), optional `wandb` / `mlflow` logging.
  - Scripts: `build.sh`, `optimize.sh` (run GEPA over logged episodes), `run.sh` for executing the optimized prompt.
  - Assets: seed system prompt, evaluation dataset, metric harness returning score plus textual feedback (LLM-as-judge or rule-based).
- **Next Steps:**
  1. Curate compact benchmark (e.g., helpdesk agents with structured tool responses) and author feedback-rich metric callbacks.
  2. Implement GEPA optimization pipeline that snapshots Pareto frontier artifacts under `var/`.
  3. Document workflow budgets, reflection LM choices, and restart strategies in README.

## 5. MCTS Prompt Agent (PromptAgent / PROMST)
- **Path:** `examples/agentic/optimization/mcts_prompt_agent`
- **Goal:** Showcase Monte Carlo Tree Search–driven prompt optimization for multi-step tasks using PromptAgent-style planning (single-task) or PROMST-style long-horizon agents (ALFWorld/WebArena).
- **Key Pieces:**
  - Dockerfile with PromptAgent or PROMST dependencies, plus optional vLLM/transformers for local inference.
  - Scripts: `build.sh`, `optimize.sh` (runs MCTS loop over bundled benchmark), `evaluate.sh` for measuring the optimized prompt, `README.md` documenting search settings.
  - Assets: starter prompt YAML, sample dataset (e.g., BIG-bench penguins or mini WebArena task), configuration templates for API keys or local models.
- **Next Steps:**
  1. Decide whether to focus on prompt-only (PromptAgent) or multi-step environment (PROMST) variant for initial release.
  2. Implement logging of search tree statistics (reward curves, best prompts) under `var/`.
  3. Provide guidance on reproducibility knobs (tree depth, expansion count, temperature) and API cost estimates.

## Cross-Cutting Tasks
- Align folder scaffolding with existing examples (YAML metadata, cache usage, `.dockerignore`).
- Update root `README.md` to introduce the Agentic section once at least one example lands.
- Consider sample evaluation harness or telemetry guidance shared across agentic directories.
- Track emerging agent-optimization toolkits (PromptWizard, AgentFlow, Trace) for potential follow-on examples or integrations once core set above is in place.
