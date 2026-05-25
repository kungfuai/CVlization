# Shared coding-agent tasks

Small, observable, end-to-end coding tasks that any agent preset in
`examples/agentic/code/` can run against. The point is **A/B
comparability**: same prompt, same input files, same verify rules —
ask any agent (opencode, pi, autogen, future ones) and you can
compare PASS/FAIL and wall time on identical inputs.

The leading underscore in `_tasks/` is a convention: this directory
has no `example.yaml`, so `cvl list` / `cvl info` skip it. It's a
shared resource, not a CVL preset.

## Task contract

Every task is a directory containing **three** things:

| File | Purpose |
|---|---|
| `PROMPT.md` | The exact natural-language ask handed to the agent. Plain text; no front-matter, no metadata. |
| `input/` | The starting filesystem state. The adapter copies this to a fresh scratch dir and runs the agent there. May be empty (e.g. for "write a new file" tasks). |
| `verify.sh` | Receives the scratch dir as `$1`. Exits 0 = PASS, non-zero = FAIL. No stdout requirements; logs land in the adapter's per-task log file. |

## Current tasks

| Task | Input | Verify |
|---|---|---|
| `fizzbuzz/` | empty | `pytest test_fizzbuzz.py` runs and all 4 cases pass |
| `lsp_rename/` | 3-file Python project (`pyproject.toml` + `src/{lib,main,util}.py` with `add_numbers`) | `add_numbers` is gone everywhere; `sum_values` appears in def + 2 imports + 2 call sites |

## Adding a task

1. Create `_tasks/<task_name>/`
2. Drop `PROMPT.md` with the natural-language ask
3. Populate `input/` with whatever starting state the agent needs
4. Write `verify.sh "$1"` that exits non-zero if the result is wrong
5. Every agent's `verify_tasks.sh` picks it up automatically — no
   per-agent wiring needed

## Adapter contract (per agent)

Each coding-agent preset has a `verify_tasks.sh` that:

- Iterates over `_tasks/*/` (skipping any without `PROMPT.md`)
- Copies `input/` to a fresh scratch dir under `${SCRATCH:-/tmp/...}`
- Runs the agent there with the prompt (headless mode, agent-specific)
- Runs the task's `verify.sh "$scratch"` and records PASS/FAIL + wall
- Prints a summary table at the end

See `examples/agentic/code/pi/verify_tasks.sh` for the reference
implementation.
