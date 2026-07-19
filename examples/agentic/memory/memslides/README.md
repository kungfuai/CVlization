# MemSlides — Hierarchical-Memory Agent (personalized slide generation)

A minimal, **CPU-viable** agentic example that distills the transferable idea from
[MemSlides](https://github.com/huohua325/Memslides)
([paper](https://huggingface.co/papers/2606.17162),
[project](https://memslides.github.io/)): a **tri-partite memory architecture**
plus **scoped slide-local revision**. The point of this example is the reusable
*memory pattern*, not the slide product — the deck is a small JSON structure so
the whole pipeline runs on CPU with no GPU, no LibreOffice, and no API key
(a deterministic mock brain is the default; real LLMs plug in via LiteLLM).

## Why

Existing CVlization agentic examples cover `rag`, `tool_use`, `long_context`,
`code`, and `web` — but none demonstrate an **explicit agent memory
architecture**. This example fills that gap with the smallest faithful version
of the MemSlides design.

## What: the three memory layers

| Layer | Lifetime | Scope | What it stores | Where |
|-------|----------|-------|----------------|-------|
| **UserProfileMemory** | long-term | per user, across decks | durable preferences: theme, tone, bullet budget, avoided topics | `users/<user_id>.json` |
| **WorkingMemory** | session | one deck-authoring session | temporary constraints + ordered edit-state across feedback turns | `sessions/<session>.json` |
| **ToolMemory** | reusable | shared | cached edit *recipes* keyed by operation signature, so repeated edits skip re-planning | `tools.json` |

All three persist as small JSON files under a host-mounted cache
(`~/.cache/cvlization/memslides/`), so a second run **reuses** learned memory
instead of re-learning it — the CPU analogue of "don't re-download the weights."

### Scoped local revision

Feedback like *"make slide 3 more concise"* is **localized** to the smallest
affected region (one slide, or a single deck-level field for a theme change) and
applied as a patch. The alternative — regenerating the whole deck — churns
unrelated slides. The `evaluate` preset quantifies the difference.

### The agent loop

```
generate  →  personalize round-0 deck from UserProfileMemory
revise     for each feedback turn:
   localize(feedback)              # Brain: mock rules or LLM
   → ToolMemory.retrieve(op-sig)   # reuse a recipe if seen before
   → apply minimal patch           # smallest affected slide region
   → WorkingMemory.record(...)     # carry constraint/edit-state forward
   → if durable ("always …"): UserProfileMemory.promote(...)
```

`memslides/agent.py` is deliberately thin: all *reasoning* lives behind the
`Brain` interface (`memslides/llm.py`), all *memory* in `memslides/memory.py`.

## Quick start

```bash
# 1. Build the CPU-only image
bash build.sh

# 2. Round-0 personalized generation (writes artifacts/deck.json)
bash generate.sh --profile data/user_profile.example.json

# 3. Multi-turn scoped local revision
bash revise.sh --feedback-file data/feedback.example.txt

# 4. Evaluation / smoke test (scoped-vs-full, memory reuse, preference adherence)
bash evaluate.sh
```

Everything above runs with the **mock brain** (no API key). To use a real LLM:

```bash
export MEMSLIDES_PROVIDER=openai        # or groq / ollama
export OPENAI_API_KEY=sk-...            # GROQ_API_KEY / OLLAMA_BASE_URL for others
bash generate.sh
bash revise.sh --feedback-file data/feedback.example.txt
```

You can also run the scripts directly (no Docker):

```bash
python generate.py --memory-dir /tmp/mem
python revise.py   --memory-dir /tmp/mem --feedback-file data/feedback.example.txt
python evaluate.py --memory-dir /tmp/mem --fresh
```

## Options

**generate.py**
- `--profile PATH` — user profile + brief JSON (seeds long-term memory on first run)
- `--topic / --audience / --sections ...` — override the brief
- `--user-id`, `--memory-dir`, `--output`
- `--provider {mock,openai,groq,ollama}`, `--model`

**revise.py**
- `--deck PATH` — input deck (default `artifacts/deck.json`)
- `--feedback "..." [...]` and/or `--feedback-file PATH`
- `--mode {local,full}` — `local` = scoped patch (default); `full` = regenerate each turn
- `--session-id`, `--user-id`, `--memory-dir`, `--output`, `--report`

**evaluate.py**
- `--fresh` — wipe the eval memory dir first
- `--memory-dir`, `--report`, `--provider`, `--model`

## Output

Your run writes to `artifacts/` (git-ignored):

- `artifacts/deck.json` / `deck.txt` — the generated deck (JSON + readable render)
- `artifacts/deck_revised.json` / `.txt` — after revision
- `artifacts/revision_report.json` — per-turn edit surface, tool reuse, promotions
- `artifacts/evaluate_report.json` — the three-check comparison

### Inspect a verified run without running anything

A committed, deterministic mock-brain bundle lives in
[`verified_run/`](verified_run/) so you can read the actual artifacts
straight from the repo:

- Round-0 personalized deck — [`verified_run/deck.txt`](verified_run/deck.txt)
  ([json](verified_run/deck.json))
- After scoped revision — [`verified_run/deck_revised.txt`](verified_run/deck_revised.txt)
  ([json](verified_run/deck_revised.json)) — theme → dark, slides 3 & 4
  shortened, a Pricing slide inserted, unrelated slides untouched
- Per-turn revision report — [`verified_run/revision_report.json`](verified_run/revision_report.json)
- Evaluation report — [`verified_run/evaluate_report.json`](verified_run/evaluate_report.json)

See [`verified_run/README.md`](verified_run/README.md) for the exact commands
that regenerate them. (These small text artifacts are committed in-repo rather
than uploaded to the `zzsi/cvl` dataset because they carry no binary media and
are deterministically reproducible from the code; a `zzsi/cvl/memslides/` mirror
is pending a dataset write token — see the PR notes.)

Example `evaluate` result (mock brain):

```
[1] scoped-vs-full     : local touches 0.75 slides/turn (preserve 0.923)
                         vs full 2.5 (preserve 0.786)                -> PASS
[2] memory reuse       : pass1 hits=1 misses=3, pass2 hits=4         -> PASS
[3] preference adherence: durable theme 'dark' honored in new deck   -> PASS
```

Interpretation: scoped revision changes **~1 slide per turn and preserves 92%**
of the deck, while full regeneration touches **2.5 slides per turn and preserves
79%**; the second pass **reuses** cached tool recipes; and a durable
*"always use a dark theme"* instruction is promoted to the profile and honored
in a brand-new, unrelated deck.

## Layout

```
memslides/
├── generate.py / revise.py / evaluate.py   # CLI entry points (presets)
├── *.sh                                     # Docker wrappers
├── memslides/                               # the reusable package
│   ├── memory.py   # UserProfileMemory / WorkingMemory / ToolMemory / MemoryStore
│   ├── deck.py     # Deck/Slide model + edit-surface diff metrics
│   ├── llm.py      # Brain interface: MockBrain (deterministic) + LLMBrain (LiteLLM)
│   └── agent.py    # MemSlidesAgent: personalize, scoped revise, full_regenerate
├── data/           # sample user profile + feedback script
└── verified_run/   # committed deterministic mock-brain artifacts (inspectable)
```

## Notes & limitations

- The **mock brain** is a genuine rule-based stand-in (topic-aware generation,
  keyword localization), not a stub — it makes the pattern reproducible and
  CI-friendly. Swap in a real model with `--provider` for higher-quality copy.
- This is intentionally *not* a pixel-perfect deck renderer. Rendering to PPTX
  is orthogonal to the memory contribution and is left out to keep the example
  CPU-only and dependency-light.

## References

- Code: https://github.com/huohua325/Memslides
- Paper: *MemSlides: A Hierarchical Memory Driven Agent Framework for Personalized
  Slide Generation with Multi-turn Local Revision* — https://huggingface.co/papers/2606.17162
- Project page: https://memslides.github.io/
