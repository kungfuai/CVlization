# opencode + Qwen3.x via local vLLM

[sst/opencode](https://github.com/sst/opencode) TUI coding agent (Claude
Code / Aider-like vibe) talking to a **locally-served** Qwen3.x model
through vLLM's OpenAI-compatible endpoint. Fully self-hosted coding loop
— no external API calls.

| | |
|---|---|
| Best for | Self-hosted agentic coding against your own GPUs |
| Agent | `opencode-ai` v1.15.10 (TUI, also supports headless `run "prompt"`) |
| Model server | Sibling `vllm` preset, detached mode |
| Default model | `vllm/Qwen/Qwen3.6-27B` (~51 GiB VRAM) |
| GPU on this side | 0 (opencode runs on CPU; the model is the GPU heavyweight) |
| Output | Edits files in the working directory in place |

## The two-command flow

```bash
# 1. Start the model server, detached, with the agentic-client flag set.
# VLLM_AGENT_DEFAULTS=1 adds --enable-auto-tool-choice (opencode sends
# tool_choice: "auto") + --tool-call-parser hermes (Qwen3 format) +
# --reasoning-parser qwen3 (so Qwen3 thinking lands in `reasoning`, not
# `content`) + raises max-model-len to 65 536 (opencode hard-codes
# max_tokens=32 000, the 8 192 default rejects it).
MODEL_ID=Qwen/Qwen3.6-27B VLLM_DETACH=1 VLLM_AGENT_DEFAULTS=1 \
  cvl run vllm serve

# Wait for it to come up (~30–90 s once weights are cached):
until curl -fsS http://localhost:8000/v1/models >/dev/null; do sleep 2; done

# 2. Launch opencode in your project dir.
cd /path/to/your/code
cvl run opencode-qwen3 run

# When done, stop the server:
cvl run vllm stop
```

> **What `VLLM_AGENT_DEFAULTS=1` does** (per the sibling `vllm` preset's
> `serve.sh`): adds `--enable-auto-tool-choice` + `--tool-call-parser
> qwen3_xml` + `--reasoning-parser qwen3` +
> `--default-chat-template-kwargs '{"enable_thinking":false}'`, and
> raises `--max-model-len` floor to 65 536. Without it, opencode hits
> `400 "auto tool choice requires --enable-auto-tool-choice..."` on
> its first request. For non-Qwen3 served models, override:
> `VLLM_TOOL_PARSER=llama3_json VLLM_REASONING_PARSER= VLLM_AGENT_DEFAULTS=1 ...`
> (empty `VLLM_REASONING_PARSER` skips that flag).

> **`qwen3_xml` vs `hermes`** — the parser choice is load-bearing.
> With `hermes`, opencode and vLLM silently disagree on tool-call
> wire-format: the model emits XML-style calls, vLLM returns them as
> plain text, opencode never sees a tool call, and the loop hangs
> indefinitely (we measured 23 min on a fizzbuzz task). With
> `qwen3_xml` the same task completes in ~22 s. Community consensus
> via vllm-project/vllm#29192 and Qwen3-Coder HF discussions.

The opencode container mounts your **current working directory** as
`/workspace` and uses `--network host` so it can reach the vLLM server
on `http://localhost:8000/v1`.

## Headless one-shot mode

```bash
cd /path/to/your/code
cvl run opencode-qwen3 run -- run "Add a test for the Foo class"
```

Anything after `--` is passed straight to the `opencode` CLI inside the
container. So `run "..."`, `-m vllm/Qwen/Qwen3-4B-Instruct ...`,
`--format json`, etc. all work.

## Choosing a model

Edit `opencode.json` (or override via the `-m` flag) to pick any model
the vLLM server is actually serving. The bundled config lists four
known-verified Qwen3 sizes — change `MODEL_ID` on `cvl run vllm serve`
to match:

| Model | VRAM | Tradeoff |
|---|---|---|
| `Qwen/Qwen3-4B-Instruct` | ~8 GB | Fast, weakest at multi-step coding |
| `Qwen/Qwen3.5-9B` | ~14 GB | Decent middle ground |
| `Qwen/Qwen3.6-27B` | ~51 GB | Recommended; best dense Qwen3 coder |
| `Qwen/Qwen3.6-35B-A3B` | ~66 GB | MoE; ~3B active, often a tradeoff |

Whichever you pick, make sure the `model:` field in `opencode.json`
matches the served name (or pass `-- run -m vllm/<full-id> "..."`).

## How the wiring works

- The opencode container runs `--network host`, so `http://localhost:8000`
  inside the container resolves to the host port the detached vLLM
  container bound (default 8000).
- `opencode.json` (mounted read-only into `~/.config/opencode/`) declares a
  custom provider via `"npm": "@ai-sdk/openai-compatible"`, with
  `baseURL` and `apiKey` pulled from env vars at run time:
  - `OPENCODE_BASE_URL` (default `http://localhost:8000/v1`)
  - `VLLM_API_KEY` (default `sk-local` — vLLM accepts anything unless
    started with `--api-key`)
- The "model" key in `opencode.json` follows the `<provider-id>/<model-id>`
  convention: `vllm/Qwen/Qwen3.6-27B`.

## Pointing at a remote vLLM (or any OpenAI-compatible) endpoint

```bash
OPENCODE_BASE_URL=https://my-vllm.internal/v1 \
  VLLM_API_KEY=sk-real-key \
  cvl run opencode-qwen3 run
```

Same env-var contract; the agent doesn't care whether the endpoint is
local or remote, as long as it speaks `/v1/chat/completions`.

## Verified end-to-end

On RTX PRO 6000 Blackwell (TP=2, BF16, CUDA graphs), `Qwen/Qwen3.6-27B`
through this preset:

```bash
mkdir /tmp/fizz && cd /tmp/fizz
cvl run opencode-qwen3 run -- run --dangerously-skip-permissions \
  "Write fizzbuzz.py with fizzbuzz(n) returning FizzBuzz/Fizz/Buzz/str(n);
   then test_fizzbuzz.py with pytest cases for 3, 5, 15, 7. Don't run pytest."
```

Result: **22 s** wall, two correct files written via the Write tool, all
four pytest cases pass independently when you `pytest test_fizzbuzz.py`.
vLLM decode throughput during the run: 35–45 tok/s.

For follow-on agent turns (Bash, Edit, Read tools), opencode's standard
TUI flow works the same way against the same `cvl run vllm serve` server
— start the TUI in your project dir, drive it like Claude Code or Aider.

## Alternate model: Qwen3-Coder

[`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
is an **older** Qwen3-base coder-tuned model (mid-2025) that ships
**non-thinking by default** and explicitly lists opencode as a
supported agent. With our `--default-chat-template-kwargs
'{"enable_thinking":false}'` flag forcing thinking off on Qwen3.6, it
no longer holds a behavioral edge for this preset — Qwen3.6's newer
training is the better default for most coding work. Use Qwen3-Coder
only if you specifically want its native 262 k context (vs Qwen3.6's
shorter default) or want to drop the reasoning-parser machinery
entirely (set `VLLM_REASONING_PARSER=` empty + omit the
`enable_thinking` kwarg).

## Notes

- **Linux-only `--network host`**: on macOS / Windows Docker Desktop,
  swap to `--add-host=host.docker.internal:host-gateway` in `run.sh` and
  set `OPENCODE_BASE_URL=http://host.docker.internal:8000/v1`.
- **No git required** in the workspace, but opencode walks up looking
  for a project root (nearest git dir or the cwd).
- opencode edits files **in place** in the mounted cwd. Keep a clean
  git state so you can `git diff` what the agent changed.
- The `opencode.json` in this preset is mounted **read-only**; opencode
  stores sessions under `~/.local/share/opencode/` inside the container,
  which is wiped when the container exits (`--rm`). If you want sessions
  to persist, mount a host dir there.

## References

- opencode docs: https://opencode.ai/docs/
- opencode providers (custom OpenAI-compatible): https://opencode.ai/docs/providers/
- Sibling `vllm` preset: `examples/generative/llm/vllm/`
