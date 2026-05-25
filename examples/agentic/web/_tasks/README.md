# Shared browser-agent tasks

Small, observable, end-to-end browser tasks that any agent preset in
`examples/agentic/web/` can run against. The point is **A/B
comparability**: same prompt, same verify rules — ask any agent
(browser-use, future stagehand / skyvern / etc.) and you can compare
PASS/FAIL and wall time on identical tasks.

The leading underscore in `_tasks/` is a convention: this directory
has no `example.yaml`, so `cvl list` / `cvl info` skip it. It's a
shared resource, not a CVL preset. Mirrors
[`examples/agentic/code/_tasks/`](../../code/_tasks/) for coding
agents — same shape, different category.

## Task contract

Every task is a directory containing:

| File | Purpose |
|---|---|
| `PROMPT.md` | The exact natural-language ask handed to the agent. Plain text; no front-matter, no metadata. |
| `verify.sh` | Receives the agent's output-artifacts dir as `$1`. Exits 0 = PASS, non-zero = FAIL. Typical checks: required artifacts exist (`agent_history.json` / `report.md` / `step_*.png`), agent's final result contains the expected value. |

Unlike the coding-agent tasks, browser tasks have **no `input/`**
directory — browser agents operate on the live web, not a local
filesystem starting state. The "input" is just the URL inside
`PROMPT.md`.

## Current tasks

| Task | What it tests | Expected result |
|---|---|---|
| `wikipedia_linux_year/` | Multi-step navigation (search box → click → extract). The answer is a stable historical fact and is **not in the prompt**, so a passing smoke proves real navigation + visual reasoning, not prompt regurgitation. | Final result contains `1991` |

## Adding a task

1. Create `_tasks/<task_name>/`
2. Drop `PROMPT.md` with the natural-language ask. Prefer:
   - A starting URL the agent can hit without auth
   - A specific, verifiable answer (a stable fact, a deterministic
     extraction)
   - "Reply with only X on a single line" so verify.sh has something
     clean to grep
3. Write `verify.sh "$1"` (where `$1` is the artifacts dir):
   - Check the expected files exist (`agent_history.json`,
     `report.md`, at least one `step_*.png`)
   - Check the agent's final result via `grep` against the
     `## Final result` section of `report.md` (most reliable: the
     adapter saves agent stdout to `agent.log` so you can grep there
     too)
   - Exit 0 on pass, non-zero with a diagnostic on fail
4. Every agent's `evaluate.sh` adapter picks it up automatically — no
   per-agent wiring needed

## Adapter contract (per agent)

Each browser-agent preset has an `evaluate.sh` (registered as the
`evaluate` preset in its `example.yaml`) that:

- Iterates over `_tasks/*/` (skipping any without `PROMPT.md`)
- For each task: creates a fresh scratch dir under `${SCRATCH:-/tmp/...}/<task_name>`
- Runs the agent in headless mode, pointing its `BROWSER_USE_OUTPUTS`
  (or equivalent) at that scratch dir
- Captures agent stdout/stderr to `<scratch>/<task>/agent.log`
- Runs the task's `verify.sh "<scratch>/<task>"` and records PASS/FAIL + wall
- Prints a summary table at the end

See [`examples/agentic/web/browser_use/evaluate.sh`](../browser_use/evaluate.sh)
for the reference implementation. The task contract is intentionally
identical in shape to the coding-agent corpus, so a future
"agentic-tasks-runner" tool could iterate over both categories the
same way.

## Stability and stickiness

Browser tasks are inherently more fragile than coding tasks because
they depend on live websites. When picking a task, prefer:

- **Wikipedia** — extremely conservative about layout; old facts
  don't change
- **httpbin.org** — explicit test/debug endpoint; documented stable
- **example.com** — IANA-maintained; trivially stable but unimpressive
- **GitHub repo pages** — UI changes occasionally but core widgets
  (stars / issues / file tree) survive

Avoid:

- News homepages (content changes hourly; layout changes monthly)
- Search engines (anti-bot measures, layout churn)
- SPAs with heavy auth flows
- Sites that ban automated traffic
