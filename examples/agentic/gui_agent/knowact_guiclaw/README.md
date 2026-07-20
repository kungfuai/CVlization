# KnowAct-GUIClaw: Self-Evolving GUI Agent

Cross-platform GUI automation agent with experience-attributable memory and a
self-evolving skill library. Built on the **Know-Route-Act-Reflect** architecture
from [KnowAct](https://github.com/HITsz-TMG/KnowAct) (MIT license).

## What it does

GUIClaw acts as a GUI subagent that can control Android, iOS, HarmonyOS, and
desktop interfaces by taking screenshots, reasoning about them with a vision
language model (VLM), and executing GUI actions (tap, swipe, type, navigate).

Key capabilities demonstrated:
- **Dry-run mode**: Plans GUI actions against a VLM without touching a device
- **Memory accumulation**: Stores experience from past tasks to improve future runs
- **Skill library**: Distills successful action sequences into reusable skills
- **Cross-platform**: Same agent framework across Android/iOS/desktop/HarmonyOS

## What to expect

| Item | Detail |
|------|--------|
| First-run download | ~955 MB Docker image (nanobot-ai + dependencies) |
| VLM requirement | External OpenAI-compatible endpoint serving a multimodal model |
| Default mode | `dry-run` (no device needed) |
| Output location | `knowact_guiclaw_output/` in your working directory |
| Output format | JSON result, trajectory logs, memory/skill snapshots |
| Runtime | Depends on VLM latency; typically 10-60 s per task in dry-run |

## Sample output

Sanitized dry-run samples are committed under [`sample_run/`](sample_run/) in
two episodes, demonstrating memory accumulation and reuse across runs.
Dry-run mode produces 1×1 placeholder screenshots (no real device), so the VLM
sees a blank screen and repeatedly presses Home — this is expected behavior
without a connected device.

### Episode 1 — baseline (clean memory)

Task: *"Open the Settings app and navigate to Wi-Fi"*
([`episode_1/`](sample_run/episode_1/))

GUIClaw starts with no prior memory. It bootstraps a conservative default
policy entry and runs the task. The VLM sees a 1×1 green screen and presses
Home on every step.

**Result** ([`result.json`](sample_run/episode_1/result.json)):
```json
{
  "task": "Open the Settings app and navigate to Wi-Fi",
  "result": {
    "parsed": { "success": false, "steps_taken": 15, "error": "max_steps_exceeded" },
    "token_usage": { "prompt_tokens": 25035, "total_tokens": 26057 },
    "duration_s": 40.2
  }
}
```

**Memory written** ([`policy.md`](sample_run/episode_1/policy.md)) — one entry:
> *Treat permission requests conservatively. Unless the current task explicitly
> requires and authorizes a permission, choose Deny, Cancel, Not now, or go
> back.*

Step 1 prompt: **1645 tokens** (includes default policy only).

### Episode 2 — with accumulated memory

Task: *"Open the Settings app and toggle Bluetooth"*
([`episode_2/`](sample_run/episode_2/))

Between episodes, a failure-avoidance policy entry is added to memory
(simulating what `enable_memory_extraction` produces from Episode 1's failed
trajectory):
> *When the screen is solid green or blank (1×1 placeholder), the device is in
> dry-run mode with no real UI. Repeatedly pressing Home will not change the
> screen. Instead, report the task as blocked using the done action.*

**Result** ([`result.json`](sample_run/episode_2/result.json)):
```json
{
  "task": "Open the Settings app and toggle Bluetooth",
  "result": {
    "parsed": { "success": false, "steps_taken": 15, "error": "max_steps_exceeded" },
    "token_usage": { "prompt_tokens": 25983, "total_tokens": 26864 },
    "duration_s": 21.2
  }
}
```

**Memory loaded** ([`policy.md`](sample_run/episode_2/policy.md)) — two entries
(original + failure-avoidance).

### Memory reuse evidence

| Metric | Episode 1 | Episode 2 | Delta |
|--------|-----------|-----------|-------|
| Policy entries | 1 | 2 | +1 (failure-avoidance) |
| Step 1 prompt tokens | 1,645 | 1,710 | **+65** (expanded policy context) |
| Total prompt tokens | 25,035 | 25,983 | +948 |

The +65 prompt token increase on Step 1 corresponds to the additional policy
entry being injected into the VLM prompt as "Advisory Policy Hints." The
GUIClaw agent loads all policy entries from `~/.guiclaw/memory/policy.md` at
the start of each run and injects them into every VLM call.

In dry-run mode, both episodes produce the same actions (press Home) because
the 1×1 green placeholder provides no visual information for the VLM to act
on. With a real device (ADB backend), the expanded memory context would
influence action selection, and successful task completions would also generate
reusable skill functions in `~/.guiclaw/skill/skills.py`.

### VLM resource usage (measured)

| VLM | vLLM flags | GPU | Baseline | Peak | Steady | Utilization |
|-----|-----------|-----|----------|------|--------|-------------|
| Qwen3-VL-8B-Instruct | `--gpu-memory-utilization 0.5 --max-model-len 4096` | RTX PRO 6000 (98 GB) | 15 MB | 50,097 MB | 48,639 MB | 0.512 |

Measured with 200 ms polling during a 15-step dry-run session. The agent
container itself uses no GPU (VLM is served externally).

## Prerequisites

An OpenAI-compatible VLM endpoint must be running. GUIClaw uses tool calling
(function calling), so vLLM must be started with `--enable-auto-tool-choice`
and a compatible `--tool-call-parser`. Example with Qwen3-VL-8B-Instruct:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.5
```

For larger models (e.g. Qwen3.5-35B-A3B), add `--tensor-parallel-size 2`.

For Android automation, connect a device or start an emulator with ADB:

```bash
adb devices  # verify device is connected
```

## Quick start

```bash
# Build
./build.sh

# Dry-run (plans actions, no device needed)
GUICLAW_BASE_URL=http://localhost:8000/v1 \
GUICLAW_MODEL=Qwen/Qwen3-VL-8B-Instruct \
./predict.sh --task "Open the Settings app and navigate to Wi-Fi"

# With Android device
GUICLAW_BACKEND=adb \
GUICLAW_BASE_URL=http://localhost:8000/v1 \
GUICLAW_MODEL=Qwen/Qwen3-VL-8B-Instruct \
./predict.sh --task "Open Settings and enable Wi-Fi"
```

## Configuration

| Environment variable | Default | Description |
|---------------------|---------|-------------|
| `GUICLAW_BASE_URL` | `http://localhost:8000/v1` | VLM API endpoint |
| `GUICLAW_MODEL` | `Qwen/Qwen3.5-35B-A3B` | Model name on the endpoint |
| `GUICLAW_API_KEY` | `sk-local` | API key for the endpoint |
| `GUICLAW_BACKEND` | `dry-run` | Backend: `dry-run`, `adb`, `local`, `ios`, `hdc` |
| `GUICLAW_MAX_STEPS` | `15` | Max GUI steps per task |
| `GUICLAW_STATE` | `~/.cache/cvlization/guiclaw` | Host dir mounted as `~/.guiclaw` for memory/skill persistence |

## Architecture: Know-Route-Act-Reflect

```
User task
   |
   v
[Know] -- retrieve memory, skills, context
   |
   v
[Route] -- decompose into subtasks, assign apps
   |
   v
[Act] -- GUIClaw observe-reason-act loop
   |        (GUI primitives + skills + shortcuts)
   v
[Reflect] -- distill trajectory into memory + skills
```

- **Know**: Gathers task context, retrieves relevant experience memory and
  candidate skills from prior runs.
- **Route**: Decomposes multi-app tasks into subtasks with typed input/output
  contracts. A blackboard mediates cross-app data flow.
- **Act**: Executes each subtask via a hybrid action space (GUI taps/swipes,
  reusable skills, Android deeplinks/intents).
- **Reflect**: Post-run summarization extracts new skills and experience
  memory that feed back into Know for future tasks.

## Memory and skill persistence

GUIClaw stores learned artifacts in `~/.guiclaw/`:

| Path | Content |
|------|---------|
| `~/.guiclaw/config.yaml` | Agent configuration |
| `~/.guiclaw/gui_runs/` | Trajectory logs and screenshots per run |
| `~/.guiclaw/memory/` | Experience memory (policy rules, lessons learned) |
| `~/.guiclaw/skill/skills.py` | Validated reusable action sequences |

These persist across runs via the `GUICLAW_STATE` host mount
(`~/.cache/cvlization/guiclaw` by default) and accumulate over episodes,
enabling the agent to improve on repeated tasks. Memory and skills are transferable across base
models (+8.5% with Kimi-2.6, +16.2% with Qwen3.5-35B-A3B on MobileWorld).

## Limitations

- **Dry-run mode** only plans actions; it does not interact with a real device.
  Use `--backend adb` with a connected Android device for full automation.
- Requires a **multimodal VLM** endpoint (pure text LLMs will not work).
- Android ADB backend needs the host to forward the ADB socket into Docker.
- iOS/HarmonyOS backends are untested in this CVlization wrapper.
- The upstream `nanobot-ai` package produces a ~955 MB Docker image.

## References

- **Paper**: [Know Deeply, Act Perfectly](https://huggingface.co/papers/2607.12625)
  (HIT Shenzhen, 2026)
- **Code**: [HITsz-TMG/KnowAct](https://github.com/HITsz-TMG/KnowAct) (MIT)
- **Result**: 64.1% on MobileWorld GUI-Only (SOTA) with open-source Kimi-2.6
