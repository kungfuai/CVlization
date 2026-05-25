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
| `wikipedia_linux_year/` | Multi-step navigation: open Wikipedia → use search box → click result → extract a stable historical fact. | Final result contains `1991` |
| `httpbin_form_fill/` | **Form interaction**: open httpbin form → fill 3 text fields → submit → read the JSON response and extract a field that wasn't in the prompt. | Final result is `https://httpbin.org/post` (the response's `url` field — different from the `/forms/post` URL the agent started on) |
| `pypi_requests_version/` | **Search + click + structured extraction**: navigate to pypi.org → search for 'requests' → click the package → extract the latest semver. | Final result matches `2.X.Y` with minor ≥ 28 |
| `github_star_comparison/` | **Multi-page comparison**: visit two GitHub repo pages, read each star count via vision, decide which is larger. | Final result contains `linux` (torvalds/linux vs microsoft/typescript — durable 2.15× gap as of 2026-05) |
| `wikipedia_python_infobox/` | **Structured JSON output**: extract three fields from the Python article's infobox into a single-line JSON object. | JSON parses + `first_appeared==1991` + `paradigm` mentions "object" + `typing_discipline` non-trivial |
| `hn_top_story_clickthrough/` | **Cross-domain research chain**: open Hacker News → identify the #1 ranked story → click its title link → report the destination URL (often a different domain). | Reports a well-formed URL with a real host, and not the HN homepage |
| `github_open_issue_count/` | **Counting + API-grounded ground truth**: open `browser-use/browser-use/issues`, read the "Open" tab's count, report as an integer. | Number within 1.5× of the GitHub search API's `is:issue is:open` count |
| `compare_pypi_github_versions/` | **Multi-source cross-verification (5+ hop, hard)**: navigate to pypi.org/project/pytest, note the version; navigate to GitHub releases, note the tag; compare and report `match:` or `mismatch:` with semvers. | Verdict line with at least one plausible semver (N.M.P, N≥7). **Requires Qwen3 thinking mode ON to pass on small VLMs** — see "Thinking mode" note below. |

All eight tasks' "answers" are derived from page content the agent
must actually navigate to and read — none of the expected values
appear in the prompts, so a passing smoke is meaningful.

## Thinking mode and the hard task

`compare_pypi_github_versions` is the empirical edge of Qwen3.5-9B's
capability — it requires holding two facts (semvers from two sites)
in working memory across 5+ navigation hops, then comparing them in
a structured output. With **thinking off** (the default for agent
loops because chain-of-thought between tool calls is mostly wasted
latency), Qwen3.5-9B truncates the final answer mid-output — we've
seen things like `match: pytest-9` where the semver never lands.

Flipping thinking back on:

```bash
# When starting vllm, override the agent-mode default to keep thinking ON:
MODEL_ID=Qwen/Qwen3.5-9B VLLM_DETACH=1 \
  VLLM_AGENT_DEFAULTS=1 VLLM_QWEN3_DISABLE_THINKING=0 \
  cvl run vllm serve
```

...lets the model scratchpad the comparison and produces a clean
`match: 9.0.3` (verified end-to-end against the actual current pytest
release on both sites).

**Tradeoff**: thinking adds ~10-30% wall-time per turn because the
model burns thousands of `reasoning` tokens before producing
`content`. For the other 7 tasks in this corpus, thinking-off is
faster and equally reliable. For multi-hop comparison / synthesis
tasks (this one, and the kind of task a future
`compare_npm_github_versions` or `synthesise_three_pages` would be),
thinking-on is worth the cost. Decide per-workload.

Latest verified `evaluate` runs on Qwen/Qwen3.5-9B (the smallest VLM
in our `vllm`-verified list):

| `VLLM_QWEN3_DISABLE_THINKING` | Result |
|---|---|
| `1` (default for agent mode) | 7/8 pass in ~250 s; `compare_pypi_github_versions` fails (truncated output) |
| `0` (thinking ON) | `compare_pypi_github_versions` passes — agent produced `match: 9.0.3` on a 94 s wall; other 7 tasks not re-verified with thinking on but expected to keep passing |

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
