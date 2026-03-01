---
name: rlm
description: Run a Recursive Language Model-style loop for long-context tasks. Uses a persistent local Python REPL and an rlm-subcall subagent as the sub-LLM (llm_query).
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - Bash
---

# rlm (Recursive Language Model workflow)

Use this Skill when:
- The user provides (or references) a very large context file (docs, logs, transcripts, code) that won't fit comfortably in a single prompt.
- You need to iteratively inspect, search, chunk, and extract information from that context.
- You can delegate chunk-level analysis to the rlm-subcall subagent.

## Mental model

- This Claude Code conversation = the root LM.
- Persistent Python REPL (`rlm_repl.py`) = the external memory / environment.
- Subagent `rlm-subcall` = the sub-LM (equivalent to `llm_query` in the RLM paper).

## Inputs

Accept these patterns from `$ARGUMENTS`:
- `context=<path>` (required): path to the large context file.
- `query=<question>` (required): what to answer.
- Optional: `chunk_chars=<int>` (default 150000) and `overlap_chars=<int>` (default 0).

If arguments are missing, ask for the context file path and query.

## Step-by-step procedure

1. **Initialise REPL state**
   ```bash
   python3 .claude/skills/rlm/scripts/rlm_repl.py init <context_path>
   python3 .claude/skills/rlm/scripts/rlm_repl.py status
   ```

2. **Scout the context**
   ```bash
   python3 .claude/skills/rlm/scripts/rlm_repl.py exec -c "print(peek(0, 3000))"
   python3 .claude/skills/rlm/scripts/rlm_repl.py exec -c "print(f'Total: {len(content):,} chars')"
   ```

3. **Choose a chunking strategy**
   - Semantic: split on markdown headers, JSON objects, log timestamps, etc.
   - Fallback: fixed character chunks (~150K chars each).

4. **Write chunks to files** (so rlm-subcall can read them)
   ```bash
   python3 .claude/skills/rlm/scripts/rlm_repl.py exec <<'PY'
   paths = write_chunks('.claude/rlm_state/chunks', size=150000, overlap=0)
   print(f'{len(paths)} chunks written')
   PY
   ```

5. **Subcall loop** — for each chunk, invoke `rlm-subcall` with:
   - The user query
   - The chunk file path
   - Any specific extraction instructions
   Collect each subagent result into buffers:
   ```bash
   python3 .claude/skills/rlm/scripts/rlm_repl.py exec -c "add_buffer('<result from subagent>')"
   ```

6. **Synthesise** — combine buffers in the main conversation to produce the final answer.
   Optionally pass all buffers to rlm-subcall once more for a coherent draft.

## Guardrails

- Do not paste large raw chunks into the main conversation.
- Use grep/peek to locate relevant excerpts; quote only what you need.
- Subagents cannot spawn other subagents — all orchestration stays here.
- Keep scratch files under `.claude/rlm_state/`.

## Alternative approaches for this context

For reproducible, non-interactive pipelines consider the Docker-based alternatives in this repo:
- `examples/agentic/long_context/rlm_needle/` — needle-in-haystack demo (any LLM API)
- `examples/agentic/long_context/rlm_doc_qa/` — document QA (any LLM API)
