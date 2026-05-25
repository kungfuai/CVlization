# browser-use + Qwen3.x VLM via local vLLM

Headless [browser-use](https://github.com/browser-use/browser-use)
agent — a Playwright-driven Chromium that reads page screenshots,
emits structured action JSON, and chains tool calls (navigate, click,
fill, extract, scroll) — pointed at a locally-served vision-language
model through vLLM's OpenAI-compatible endpoint. Sibling to the
`agentic-opencode` / `agentic-pi` coding presets, but operates on the
**live web** instead of the local filesystem.

| | |
|---|---|
| Best for | Self-hosted browser automation; form filling; data extraction from real pages |
| License | MIT (browser-use), Apache-2.0 (Playwright), BSD-3 (Chromium) |
| Agent | `browser-use==0.12.8` (pinned) |
| Browser | Chromium via Playwright (headless by default) |
| Model server | Sibling `vllm` preset, detached, with **a VLM** |
| Default model | `Qwen/Qwen3.5-9B` (VLM, ~14 GB VRAM) |
| GPU on this side | 0 (browser-use runs on CPU; the VLM is the GPU heavyweight) |
| Image size | ~2 GB (python:3.12-slim + Playwright Chromium + deps) |

## VLM required — text-only LLMs do not work

This is the most important caveat. From browser-use's own
[`examples/models/qwen.py`](https://github.com/browser-use/browser-use/blob/main/examples/models/qwen.py)
(their words, not ours):

> "so far we only had success with **qwen-vl-max**. Other models, even
> qwen-max, do not return the right output format. They confuse the
> action schema. E.g. they return `actions: [{"navigate": "google.com"}]`
> instead of `[{"navigate": {"url": "google.com"}}]`"

browser-use sends a screenshot at every step (`use_vision=True`) and
expects the LLM to emit specific structured action JSON. **A text-only
LLM is the wrong tool for this preset.** Our default
`Qwen/Qwen3.5-9B` is the smallest VLM we've verified end-to-end in the
sibling `vllm` preset — it is at the *low end* of what works for
browser automation. Expect:

- Trivial tasks (single-page fill-and-submit, single-page extract):
  reliably work
- Multi-step navigation (3+ page hops, dynamic content): often
  confused
- Complex sites (auth flows, infinite scroll, JS-heavy SPAs):
  unreliable

If you want browser-use's full capability story, swap in `qwen-vl-max`
(Alibaba Cloud), `claude-sonnet-4-6`, or `gpt-5` — see the "Pointing
at a remote API" section below. We document this honestly because
"local open VLMs rival hosted models on browsing" is not a claim we
can make today.

## The two-command flow

```bash
# 1. Start the model server, detached. MUST be a VLM (use_vision=True
# requires it). Same VLLM_AGENT_DEFAULTS=1 the coding presets use.
MODEL_ID=Qwen/Qwen3.5-9B VLLM_DETACH=1 VLLM_AGENT_DEFAULTS=1 \
  cvl run vllm serve
until curl -fsS http://localhost:8000/v1/models >/dev/null; do sleep 2; done

# 2. Run the agent. Default task: fill httpbin.org/forms/post and report
# what the response echoes back.
cvl run agentic-browser-use run

# Or a custom task:
cvl run agentic-browser-use run -- \
  "Go to https://news.ycombinator.com and report the title of the top story"

# Cleanup:
cvl run vllm stop
```

## What browser-use can do (and not)

| Capability | Works |
|---|---|
| Navigate URLs | ✅ |
| Click visible elements | ✅ |
| Fill text inputs, select dropdowns, check radios/boxes | ✅ |
| Submit forms | ✅ |
| Extract text / scrape | ✅ |
| Take and reason about screenshots | ✅ (per step) |
| Multi-tab / new windows | ✅ |
| Persist cookies / login state across runs | ⚠ — possible with `--storage-state` patterns; we don't wire that in |
| Solve CAPTCHAs | ❌ |
| Authenticated flows that require 2FA | ❌ (you'd hand-jam the auth, then pass storage state) |
| File downloads | ⚠ — works, but lands inside the container at `/tmp/`; needs an explicit bind mount to surface |

## Configuration

Override via env vars on `cvl run agentic-browser-use run`:

| Env | Default | Purpose |
|---|---|---|
| `OPENCODE_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible chat completions endpoint |
| `VLLM_API_KEY` | `sk-local` | Any non-empty string; vllm ignores it unless started with `--api-key` |
| `BROWSER_USE_MODEL` | `Qwen/Qwen3.5-9B` | Served model id (must match vllm's `--served-model-name`) |
| `BROWSER_USE_MAX_STEPS` | `20` | Hard cap on agent reasoning steps per task |

Pass-through args (everything after `--`) go to `agent.py`:

```bash
# Custom task as positional arg
cvl run agentic-browser-use run -- "Go to example.com and tell me the H1"

# Different model + lower step cap
BROWSER_USE_MODEL=mistralai/Ministral-3-8B-Instruct-2512 \
BROWSER_USE_MAX_STEPS=10 \
  cvl run agentic-browser-use run

# Show what the agent is thinking by raising step count
cvl run agentic-browser-use run -- --max-steps 50 "complex multi-step task"
```

## Pointing at a remote VLM (Anthropic / OpenAI / Alibaba Cloud)

browser-use's `ChatOpenAI` is happy to talk to any OpenAI-compatible
endpoint. Same env-var contract:

```bash
# OpenAI / GPT-5
OPENCODE_BASE_URL=https://api.openai.com/v1 \
VLLM_API_KEY=sk-yourrealkey \
BROWSER_USE_MODEL=gpt-5-mini \
  cvl run agentic-browser-use run

# Alibaba Cloud (the qwen-vl-max path upstream actually verifies)
OPENCODE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
VLLM_API_KEY=$ALIBABA_CLOUD_API_KEY \
BROWSER_USE_MODEL=qwen-vl-max \
  cvl run agentic-browser-use run
```

For Anthropic Claude, swap `agent.py`'s `ChatOpenAI` for browser-use's
`ChatAnthropic` (see `examples/models/claude-4-sonnet.py` upstream).

## How the wiring works

- **Headless Chromium**: `BrowserProfile(headless=True)` plus
  `IN_DOCKER=1` env (set in the Dockerfile) makes browser-use disable
  the Chromium sandbox automatically — required for running inside
  containers. `--shm-size=1g` on `docker run` prevents Chromium from
  crashing on small `/dev/shm`.
- **Vision**: `Agent(..., use_vision=True)` sends a screenshot every
  step alongside the DOM. The VLM has to interpret both.
- **Schema loosening**: `ChatOpenAI(..., add_schema_to_system_prompt=True,
  remove_min_items_from_schema=True, remove_defaults_from_schema=True)`
  trades some action-schema strictness for compatibility with smaller
  open-weight VLMs that don't perfectly honour `response_format`. With
  GPT-5 / Sonnet you can drop these.
- **Network**: `--network host` so `http://localhost:8000` inside the
  container reaches the host's vLLM port (Linux-only).
- **No GPU on this side**: browser-use itself is CPU-bound (Chromium
  rendering happens in software headlessly). All the GPU work is on
  the vllm side.

## Verified end-to-end smoke

The default task (no args) is a **multi-step Wikipedia lookup** that
exercises navigate + search + click + extract on a real site:

```
Open https://en.wikipedia.org. Use the search box at the top of
the page to search for 'Linux'. Click the first result. From the
article, find and report the year that Linux was first released.
Reply with only the four-digit year on a single line.
```

**Verified result** (RTX PRO 6000, Qwen3.5-9B, ~38 s wall):

- Final result: `1991`
- 4 agent steps: `navigate → input "Linux" → click first result → done(1991)`
- 3 saved screenshots (1920×1080) showing each navigation step
- `agent_history.json` (full structured trace)
- `report.md` (markdown summary with task, actions, embedded screenshots)

All artifacts land in `./browser_use_outputs/` on the host (configurable
via `BROWSER_USE_OUTPUTS=<path>`). The answer "1991" is **not in the
prompt** — verifying it requires real navigation + visual reasoning +
extraction, so a passing smoke is meaningful.

### Other tasks worth trying (uses agent.py as-is)

```bash
cvl run agentic-browser-use run -- \
  "Go to https://github.com/browser-use/browser-use and report the current star count."

cvl run agentic-browser-use run -- \
  "Compare the star counts of github.com/browser-use/browser-use and \
   github.com/microsoft/playwright; report which has more."

cvl run agentic-browser-use run -- \
  "Go to https://news.ycombinator.com; list the titles of the top 3 stories."
```

Multi-step tasks beyond ~5-6 hops degrade quickly on Qwen3.5-9B — see
the VLM caveat at the top of this README and the upgrade path below.

## Notes

- **Linux-only `--network host`**: on macOS / Windows Docker Desktop,
  swap to `--add-host=host.docker.internal:host-gateway` in `run.sh`
  and set `OPENCODE_BASE_URL=http://host.docker.internal:8000/v1`.
- **VRAM on the model side**: Qwen3.5-9B is ~14 GB at bf16, so a 16+
  GB GPU is the realistic floor for the default model. The browser
  side adds nothing.
- **No persistent browser state**: the container is `--rm`'d each run;
  cookies / cache / storage state are wiped. If you need session reuse
  (e.g. for a logged-in flow), mount a host dir into Playwright's
  user-data-dir.
- **Telemetry**: browser-use sends anonymous usage stats via posthog by
  default. Set `BROWSER_USE_ANALYTICS=false` on `cvl run` to opt out.

## References

- browser-use: https://github.com/browser-use/browser-use
- browser-use docs: https://docs.browser-use.com
- Playwright Python: https://playwright.dev/python/
- Sibling `vllm` preset (`VLLM_AGENT_DEFAULTS=1` recipe): `examples/generative/llm/vllm/`
- Sibling coding agents: `examples/agentic/code/{opencode,pi}/`
