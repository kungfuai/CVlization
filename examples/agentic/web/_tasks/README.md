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
| `arxiv_paper_title/` | **PDF reading via the agent's read_file tool**: navigate to `arxiv.org/pdf/1706.03762`, recognize the page is a PDF viewer (browser-use auto-downloads PDFs), call the `read_file` action on the downloaded path, extract the paper title. | Final result contains `Attention Is All You Need`. **Requires Qwen3 thinking mode ON on small VLMs** — without thinking, the agent can't follow the system reminder about routing to `read_file` instead of `extract`. |
| `python_docs_function_lookup/` | **Practical API lookup**: open the official Python `os` module docs, find the function that returns the current working directory, report its name. | Final result contains `getcwd` (or `os.getcwd`). |
| `llm_history_research_brief/` | **Deep research with long structured output (hardest)**: visit three Wikipedia articles (Transformer / BERT / GPT-3), extract year + org + parameter count for each, then synthesize a markdown report with a title, intro, comparison table, AND a "Common themes" synthesis paragraph. | Markdown structure (H1 + table + `## Common themes`), required facts (`2017` / `2018` / `2020` / `Google` / `OpenAI` / `175B`), length 400-5000 chars. **Requires a bigger VLM to pass** — Qwen3.5-9B reliably visits all three sources and emits the table but consistently treats the table as the endpoint and never writes the synthesis paragraph, regardless of thinking mode or temperature (tested both). See "Three tiers" below. |

All eight tasks' "answers" are derived from page content the agent
must actually navigate to and read — none of the expected values
appear in the prompts, so a passing smoke is meaningful.

## Three tiers, by model size required

After empirical testing on Qwen3.5-9B (our default VLM, the smallest
verified in the sibling `vllm` preset), the corpus naturally splits
into three tiers:

| Tier | Tasks | Model required | Pass condition |
|---|---|---|---|
| **Easy** (8) | wiki_linux_year, httpbin_form_fill, pypi_requests_version, github_star_comparison, wiki_python_infobox, hn_top_story_clickthrough, github_open_issue_count, python_docs_function_lookup | Qwen3.5-9B, thinking OFF | direct extract-and-report; structured output is short |
| **Medium** (2) | compare_pypi_github_versions, arxiv_paper_title | Qwen3.5-9B, **thinking ON** | model needs scratchpad to follow nuanced multi-step recipe (5+ hop comparison; PDF read_file routing) |
| **Hard** (1) | llm_history_research_brief | **bigger VLM** (Claude Sonnet / GPT-5 / qwen-vl-max via `OPENCODE_BASE_URL` override) | long structured synthesis after multi-source extraction; Qwen3.5-9B reliably produces the structured *table* portion but truncates before the synthesis paragraph |

## Thinking mode and the medium-tier tasks

Two tasks in this corpus are at the empirical edge of Qwen3.5-9B's
capability and only pass with **thinking mode ON**:

- `compare_pypi_github_versions` — holds two semvers in working memory
  across 5+ navigation hops, then emits a structured comparison.
  Without thinking, the model truncates mid-output (e.g. `match: pytest-9`
  instead of `match: 9.0.3`).
- `arxiv_paper_title` — the agent must recognize a PDF page, ignore
  the browser-use system reminder *not* to use `extract` on it, and
  instead route to the `read_file` action with the downloaded PDF
  path. Without thinking, the model emits three consecutive
  "format error" actions and gives up.

Both pass cleanly with thinking on. Flipping it back on at server
start:

```bash
MODEL_ID=Qwen/Qwen3.5-9B VLLM_DETACH=1 \
  VLLM_AGENT_DEFAULTS=1 VLLM_QWEN3_DISABLE_THINKING=0 \
  cvl run vllm serve
```

**The pattern**: thinking-on works for any task that requires the
model to follow a nuanced multi-step recipe between observation and
action. Thinking-off (the default for `VLLM_AGENT_DEFAULTS=1`) is
fine for direct extract-and-report tasks; faster too.

**Tradeoff**: thinking adds ~10-30% wall-time per turn because the
model burns thousands of `reasoning` tokens before producing the
`content` browser-use actually consumes. For the 8 "easy" tasks in
this corpus, thinking-off is faster and equally reliable. For the 2
"hard" tasks (compare_*, arxiv_*), thinking-on is the unlock.

Verified `evaluate` results on Qwen/Qwen3.5-9B (the smallest VLM in
our `vllm`-verified list):

| `VLLM_QWEN3_DISABLE_THINKING` | Result |
|---|---|
| `1` (default for agent mode) | 8/10 pass; `compare_pypi_github_versions` and `arxiv_paper_title` fail honestly (small-model instruction-following ceiling) |
| `0` (thinking ON) | All hard tasks verified to pass individually: `compare_pypi_github_versions` → `match: 9.0.3` (94 s), `arxiv_paper_title` → `Attention Is All You Need` (64 s). The other 8 should keep passing (thinking-on doesn't break behaviour, just slows it). |

**Why thinking unblocks these specifically**: the failure mode in
both cases was instruction-following, not vision or extraction. The
model in both cases had all the information it needed (saw both
pages, or saw the PDF download notification) but couldn't reason
through the right action format / structured output. Thinking gives
it a scratchpad to plan the action before emitting it.

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
