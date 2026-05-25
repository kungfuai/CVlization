# pi (`omp`) + Qwen3.x via local vLLM

[oh-my-pi](https://github.com/can1357/oh-my-pi) (`omp`) — a fork of
[mariozechner/pi-mono](https://github.com/mariozechner/pi-mono) — is a
coding TUI with **the IDE wired in**: LSP-aware edits (rename actually
walks re-exports + barrels via `workspace/willRenameFiles`), DAP
debugger attaches (lldb / dlv / debugpy), persistent Python + Bun
execution kernels with bidirectional tool calls, first-class subagents,
and a 32-tool surface. This preset wires it to a locally-served Qwen3.x
model via the OpenAI-compatible endpoint of our sibling `vllm` preset.

| | |
|---|---|
| Best for | Self-hosted agentic coding with LSP + debugger integration |
| Agent | `@oh-my-pi/pi-coding-agent` (binary: `omp`), built from `can1357/oh-my-pi@3b072a10` |
| Model server | Sibling `vllm` preset, detached mode |
| Default model | `vllm/Qwen/Qwen3.6-27B` (~51 GiB VRAM) |
| GPU on this side | 0 (omp runs on CPU; the model is the GPU heavyweight) |
| Image size | ~2-4 GB (Rust + Bun + Python + native addon + bundled language servers) |

## How it differs from the sibling `opencode` preset

| | `opencode` | **`pi`** |
|---|---|---|
| Tool surface | basic Write / Edit / Bash / Read | **32 built-in tools, 13 LSP ops, 27 DAP ops** |
| LSP integration | none | rename routes through `workspace/willRenameFiles` |
| Debugger | none | attaches lldb / dlv / debugpy via DAP |
| Code execution | bash tool only | persistent Python + Bun kernels with bidirectional tool calls |
| Subagents | n/a | first-class, schema-validated returns |
| Web search | basic | 14-provider chained search |
| Image | 1.83 GB | ~2-4 GB |
| Velocity | weekly | very fast (v15.x) |

Use opencode when you want the lean, simple TUI; use pi when you want
language-server-aware refactors, a debugger loop, or persistent code
execution.

## The two-command flow

```bash
# 1. Start the model server, detached, with the agentic-client flag set.
# VLLM_AGENT_DEFAULTS=1 wires the four flags every agent client needs
# (tool-choice / parser / reasoning-parser / enable_thinking off) plus
# raises max-model-len to 65 536. Same defaults the opencode preset uses.
MODEL_ID=Qwen/Qwen3.6-27B VLLM_DETACH=1 VLLM_AGENT_DEFAULTS=1 \
  cvl run vllm serve

# Wait for it to come up (~30-90 s once weights are cached):
until curl -fsS http://localhost:8000/v1/models >/dev/null; do sleep 2; done

# 2. Launch pi in your project dir.
cd /path/to/your/code
cvl run pi run                                      # TUI

# When done, stop the server:
cvl run vllm stop
```

## Headless one-shot mode

```bash
cd /path/to/your/code
cvl run pi run -- -p "Add a test for the Foo class"
```

The opencode equivalent of `run "prompt"` is `-p "prompt"` (or
`--print "prompt"`) in pi. Anything after `--` is passed straight to
the `omp` CLI. So `--model vllm/Qwen/Qwen3-4B-Instruct`, `--tools …`,
etc. all work.

## Using pi on your own code

The container bind-mounts your **current working directory** to
`/work` and `--workdir`s into it. So everything pi reads / writes /
edits / runs happens against the real files on your host filesystem.

### What pi can do on your code

| Operation | Works? | Notes |
|---|---|---|
| **Read files** (`tool.read`) | ✅ | Recursive; follows symlinks within the mount |
| **Write new files** | ✅ | Verified by the `fizzbuzz` task |
| **Edit existing files** | ✅ | `strReplace` + diff-style edits |
| **Run Bash in your repo** | ✅ | `git status`, `pytest`, `npm test` — they hit your real repo |
| **Grep / search** | ✅ | Bundled Rust-native ripgrep-style tool |
| **LSP rename / refs / diagnostics** | ✅ * | Needs a project root marker (`pyproject.toml` / `setup.py` / `requirements.txt` / `Pipfile` for Python) — see LSP section above |
| **Run pytest / build / lint** | ✅ | As a Bash invocation |
| **Web search + fetch** | ✅ | pi's `web_search` chains 14 providers |
| **Persistent Python / Bun cells** | ✅ | Data analysis on files in your repo |

### Recipe — pi on a real codebase

```bash
# One-time, ~10 min first build (Rust + Bun + Python wheel + native addon)
cvl run agentic-pi build

cd ~/my_real_codebase
git status      # make sure you have a clean tree -- pi edits in place

# Terminal 1: model server (with the four agent flags VLLM_AGENT_DEFAULTS sets)
MODEL_ID=Qwen/Qwen3.6-27B VLLM_DETACH=1 VLLM_AGENT_DEFAULTS=1 \
  cvl run vllm serve
until curl -fsS http://localhost:8000/v1/models >/dev/null; do sleep 2; done

# Terminal 2: pi against your repo
cvl run agentic-pi run                                # interactive TUI
# or headless, one prompt at a time:
cvl run agentic-pi run -- -p "Read main.py and tell me what it does"
cvl run agentic-pi run -- -p "Add type hints to all functions in src/utils.py"
cvl run agentic-pi run -- -p "Rename function bar to baz across the repo via lsp"

git diff        # inspect what pi changed before keeping
cvl run vllm stop    # tear down the model server when you're done
```

### Gotchas that will bite first-time users

- **Files written by pi are owned by `root`.** Inside the container pi
  runs as root, so new files / edits land on your host as root-owned.
  Either run `sudo chown -R $USER:$USER .` after the session, or set
  `--user $(id -u):$(id -g)` on `docker run` in `run.sh` (we don't by
  default — root keeps the bundled language servers' caches simple).
- **pi edits in place; there is no undo.** A `git stash` or
  `git commit -am 'pre-pi'` before the session is cheap insurance.
  Use `git diff` after every prompt to see what changed.
- **Sessions are wiped on container exit.** pi's session DB lives at
  `/root/.omp/` inside the container; `--rm` clobbers it. Multi-turn
  context across separate `cvl run agentic-pi run` invocations does
  not persist. Mount a host dir to `/root/.omp/` if you need it.
- **The TUI needs a real terminal.** Works in tmux, iTerm, standard
  terminal emulators. Doesn't work over `ssh -T` or from a
  non-interactive script — use the headless `-- -p "prompt"` path
  instead.
- **Cwd silently matters.** `cvl run agentic-pi run` from `/tmp` makes
  pi happily operate on `/tmp`. No "this doesn't look like a project"
  check today.
- **The vllm sidecar doesn't auto-stop.** After your `agentic-pi run`
  exits, the detached vllm server is still up. Run `cvl run vllm stop`
  or you'll leak a multi-GiB container.

## How the wiring works

- **Build path** — pi has no prebuilt Docker image. `build.sh` clones
  `can1357/oh-my-pi` into `~/.cache/cvlization/oh-my-pi`, builds the
  upstream `oh-my-pi/pi:dev` runtime image (~5-10 min first build:
  Rust + Bun + Python wheel), then layers our derived image on top
  with `python-lsp-server` and `typescript-language-server` for the
  LSP tool to drive. Override the upstream commit with
  `PI_REPO_REF=<sha-or-branch>`.
- **Model config** — pi reads `~/.omp/agent/models.yml`. Our
  `run.sh` bind-mounts a generated copy of `models.yml` (from this
  preset's directory, with `baseUrl` patched from
  `OPENCODE_BASE_URL`) into the container, declaring a custom
  `vllm` provider with four Qwen3.x model entries.
- **Network** — `--network host` so `http://localhost:8000` inside
  the container reaches the host vLLM port (Linux only; on macOS use
  `--add-host=host.docker.internal:host-gateway`).
- **Auth** — vLLM accepts any string. We default `VLLM_API_KEY=sk-local`
  and have pi's `apiKey:` reference that env var name (pi resolves
  `apiKey: VLLM_API_KEY` by first checking `Bun.env.VLLM_API_KEY`).
- **`disableStrictTools: true`** on the provider — vLLM's OpenAI-compat
  tool-call format doesn't always pass pi's default strict check; this
  flag accepts the Qwen3-flavoured calls the `--tool-call-parser
  qwen3_xml` flag produces on the vLLM side.

## LSP

pi's `lsp` tool is enabled by default. The bundled `models.yml` ships
with the corresponding language servers installed in the image:

- **Python** — `python-lsp-server[all]` (`pylsp`)
- **TypeScript** — `typescript-language-server` + `typescript`

For other languages, add the relevant server to the Dockerfile (e.g.
`rust-analyzer`, `gopls`, `clangd`). pi's defaults map about 15
languages to their canonical servers — see
[`packages/coding-agent/src/lsp/defaults.json`](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/src/lsp/defaults.json)
in the upstream repo.

You don't invoke LSP ops by slash command; you ask the agent in plain
English ("rename `foo` to `bar` in `src/lib.ts`") and the `lsp` tool
is one of the things it chooses to call.

> **Project root marker required.** pi's Python LSP entries
> (`pyright`, `basedpyright`, `pylsp`) all expect a `pyproject.toml`,
> `setup.py`, `requirements.txt`, or `Pipfile` in the project root
> before the language server will start. A bare `src/*.py` tree won't
> trigger LSP; the agent will correctly report it can't find a
> configured Python server (and may burn several minutes thinking about
> why before giving up). Add a minimal `pyproject.toml` (just
> `[project]` + `name` + `version`) and rename / references /
> diagnostics light up.

### LSP rename — verified example

With `pyproject.toml` present, ask:

> "Use the lsp tool to rename `add_numbers` in `src/lib.py` to
> `sum_values`. Propagate to all callers (apply=true). Don't use
> Write or Edit."

pi calls `lsp(action="rename", file=..., symbol=..., new_name=...,
apply=true)`, pylsp returns a workspace edit covering the definition
site plus every `from .lib import add_numbers` and every call site;
the agent applies the edit and reports back. Verified end-to-end:
**26 s wall** on a 3-file project, 5 sites renamed in one tool call.

## Evaluating — `evaluate` preset

A repeatable evaluation runner is wired in. It iterates over every
task in [`../_tasks/`](../_tasks/) (currently `fizzbuzz/` and
`lsp_rename/`), runs pi against each, and reports per-task PASS / FAIL
with wall time. Naming matches the sibling
`autogen_pair_programmer`'s `evaluate` preset.

```bash
# vllm must already be running:
MODEL_ID=Qwen/Qwen3.6-27B VLLM_DETACH=1 VLLM_AGENT_DEFAULTS=1 \
  cvl run vllm serve

cvl run agentic-pi evaluate
# ...
# === summary ===
#   [fizzbuzz]   PASS  wall=32s
#   [lsp_rename] PASS  wall=26s
#   2/2 passed in 59s
```

The corpus in `examples/agentic/code/_tasks/` is shared across every
coding agent preset under `examples/agentic/code/` — A/B comparing
agents on identical inputs is just running each preset's `evaluate`.
See `../_tasks/README.md` for the task contract and how to add a new
task.

## Choosing a model

`models.yml` lists four Qwen3 sizes. Pi's `--model` flag is fuzzy:

```bash
cvl run pi run -- --model vllm/Qwen3-4B-Instruct
# or:
PI_MODEL=vllm/Qwen/Qwen3.6-35B-A3B cvl run pi run
```

Match the served model on the vllm side via `MODEL_ID=`. To talk to a
served model not in our bundled list, edit `models.yml` (or pass
`--model <provider>/<id>` and let fuzzy match find it).

## Pointing at a remote vLLM (or any OpenAI-compatible) endpoint

```bash
OPENCODE_BASE_URL=https://my-vllm.internal/v1 \
  VLLM_API_KEY=sk-real-key \
  cvl run pi run
```

`run.sh` rewrites the `baseURL` in the mounted `models.yml` from
`OPENCODE_BASE_URL`. Same env-var contract as the sibling opencode
preset.

## Notes

- **Linux-only `--network host`**: on macOS / Windows Docker Desktop,
  swap to `--add-host=host.docker.internal:host-gateway` in `run.sh`
  and set `OPENCODE_BASE_URL=http://host.docker.internal:8000/v1`.
- pi stores **sessions** under `/root/.omp/` inside the container.
  Wiped on `--rm`. Mount a host dir there if you need persistent
  history.
- The MIT-licensed upstream (Mario Zechner + Can Bölük) ships its own
  Dockerfile (`/tmp/cvl/oh-my-pi/Dockerfile`); we extend `pi-runtime`
  (its self-contained target) rather than re-implementing from scratch.

## References

- oh-my-pi: https://github.com/can1357/oh-my-pi
- Original Pi: https://github.com/mariozechner/pi-mono
- Sibling `opencode` preset: `examples/agentic/code/opencode/`
- Sibling `vllm` preset (`VLLM_AGENT_DEFAULTS=1` recipe): `examples/generative/llm/vllm/`
