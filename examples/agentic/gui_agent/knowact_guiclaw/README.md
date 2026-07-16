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

## Prerequisites

An OpenAI-compatible VLM endpoint must be running. For example, serve
Qwen3.5-35B-A3B with vLLM:

```bash
vllm serve Qwen/Qwen3.5-35B-A3B --tensor-parallel-size 2
```

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
GUICLAW_MODEL=Qwen/Qwen3.5-35B-A3B \
./predict.sh --task "Open the Settings app and navigate to Wi-Fi"

# With Android device
GUICLAW_BACKEND=adb \
GUICLAW_BASE_URL=http://localhost:8000/v1 \
GUICLAW_MODEL=Qwen/Qwen3.5-35B-A3B \
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

These persist across runs and accumulate over episodes, enabling the agent to
improve on repeated tasks. Memory and skills are transferable across base
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
