# CVlization: Repo Strategy & Recommendations

## 0) One‑liner Positioning

**A curated, dockerized example garden + tiny CLI** that lets busy ML engineers **Run → Compare → Copy** reproducible recipes into their own repos—without adopting a new framework.

---

## 1) Who It Serves & When It's Useful

* **Consultants & staff ML engineers** who need a working baseline/demo **in minutes**.
* **Researchers** who want apples‑to‑apples comparisons under identical infra.
* **Teams** who plan to **copy** a proven slice into their org repo and keep moving.

**They'll use CVlization when it delivers:**

* **TTFE ≤ 5 min** (one command to train/eval/serve).
* **Apples‑to‑apples** metrics across variants.
* **Exportability**: turnkey copy‑out with pins and CI.

---

## 2) Repo Shape (High‑Level)

* **Examples‑first monorepo** (each example fully self‑contained & pinned).
* **Tiny core**: a *protocol + CLI*, not a training framework.
* **Aggressive pruning & graduation**: remove bitrotted examples, export successful ones as standalone repos.

```
/                 # mono index, docs, CLI
  tools/         # cvl CLI (Typer/Rich) – no heavy ML deps
  core/          # tiny helpers (cache/env/record schema) – optional import
  examples/
    <task>/<ex>/ # Dockerfile, example.yaml, scripts, README (10-15 curated examples)
  legacy/        # archived cvlization/torch, cvlization/tensorflow + old examples
  .devcontainer/ # one‑click open in VS Code/Codespaces
```

---

## 3) Core vs. Examples (Scope Boundary)

**Core (keep tiny):**

* **Protocols** (no runtime coupling):

  * `example.yaml` descriptor ✅ **already exists in 97% of examples** (task/datasets/tags/resources/presets/files).
  * **Run‑record JSON** schema (params, env, metrics, timings, artifacts).
* **CLI** (`cvlz` via `pipx`):

  * `list | info | run | compare | export | doctor`.
  * Cache strategies: `--cache host | volume` (HF/Torch reuse vs fastest IO).
  * Compose orchestration from anywhere (resolves repo root).
* **Tiny utilities** (optional import): cache path resolution, env wiring, run‑record writer/validator.

**Examples (own the ML logic):**

* Training/eval/serve code, datasets, metrics, augmentation, schedulers, etc.
* Pinned environments (Dockerfile + lockfiles).
* Presets: `train`, `eval`, `serve`, `notebook`, plus compare‑friendly variants.

**Avoid in core:** abstract trainers, callback hierarchies, shared heavy deps, cross‑example base classes.

---

## 4) User Experience (CLI Contracts)

```bash
cvl list                      # discover examples w/ tags & stability
cvl info video/minisora       # GPU/VRAM/ETA, dataset, last-green badge
cvl run  nlp/bert --preset train --cache host --param lr=3e-5 --param epochs=3
cvl run  vis/resnet --preset eval --profile gpu
cvl compare runs/a.json runs/b.json --key val/accuracy
cvl export  vis/resnet --as-repo  # minimal standalone repo w/ pins + CI
cvl doctor                    # docker/nvidia/cache/disk checks
cvl cache info                # show cache sizes and locations
```

**Presets** provide contrastive comparisons (e.g., baseline vs policy aug vs mixing; LoRA vs full fine‑tune; CPU vs GPU).

---

## 5) Caching & Data Reuse

**Default:** Repo-scoped cache (`./data/container_cache` → container `/cache`) for isolation & reproducibility.

**Opt-in modes:**
* `--cache=host`: Mount `~/.cache` to reuse models downloaded outside Docker (saves disk space, faster cold starts)
  * ⚠️ Warning displayed about non-determinism
  * Requires running as host user to avoid permission issues
* `--cache=volume`: Named Docker volume (macOS performance optimization)

**Always run containers as host user** (`--user $(id -u):$(id -g)`) to avoid root-owned files.

Respect official envs: `HF_HOME=/cache/huggingface`, `TORCH_HOME=/cache/torch`.

---

## 6) Trust & Reproducibility

* **Scheduled CI smoke builds** (weekly): build images + 30‑sec CPU check per example.
* **Badges**: last‑green date, GPU required/optional, stability tier (Stable/Beta/Experimental).
* **Run‑record JSON** written on each run → source of truth for `cvlz compare`.

---

## 7) Docs as a Gallery

* Auto‑generate a searchable gallery (MkDocs) from `example.yaml` + READMEs.
* Filters: task, modality, GPU/VRAM, ETA, stability; quick copy buttons for `cvlz` commands.

---

## 8) On Legacy Core Library

* **Move `cvlization/torch` and `cvlization/tensorflow` to `/legacy` immediately.**
* Keep only `cvlization/specs/` (protocols) in core.
* Archive bitrotted examples to `/archived-examples/` with explanation.
* One modern reference loop each for PyTorch and Keras (minimal, shows run-record generation only).

---

## 9) What Not To Build

* Another end‑to‑end training framework.
* Cross‑example base classes that create version matrices.
* Heavy runtime deps inside core.

---

## 10) Decision Rubric (What belongs in core?)

1. Duplicated in ≥3 examples **and** stable?
2. Can be a **pure helper/protocol** (files/env/json) vs. a base class?
3. Won’t force core to pin heavy ML libs?
4. If removed, replacing it is ≤100 LOC copy?
5. Materially lowers TTFE or speeds Compare/Export?

Only if **yes** across the board → extract to core.

---

## 11) Graduation Criteria (When to Export Examples)

An example should graduate to its own repo when ANY of:
* Has **>3 external users** (community adoption signals value)
* Generates **>500MB artifacts** in monorepo (data, checkpoints, outputs)
* Requires **>2 breaking changes per quarter** to stay current (high maintenance)
* Becomes a **product/service** (not just a recipe to copy)

Graduation is SUCCESS, not failure—it proves the example's value and reduces monorepo burden.

---

## 12) Roadmap (30/60/90) - REVISED

**30 days - Triage & Export First**

* Audit all 73 examples: test builds, tag Green/Yellow/Red.
* Archive Red examples to `/archived-examples/` with explanation.
* Move `cvlization/torch` and `cvlization/tensorflow` to `/legacy`.
* Build minimal `cvl export --as-repo` (Dockerfile + pins + basic CI).
* Graduate 3-5 active examples (those with >500MB artifacts or active external users).

**60 days - Stabilize Core Set**

* Identify 10-15 Green examples worth keeping in monorepo.
* ✅ Add `example.yaml` to examples (**67/69 complete**, 97% coverage).
* Add run-record JSON schema and writer to examples.
* ✅ Build `cvl list` and `cvl info` for discovery (leverage existing example.yaml).
* ✅ Build `cvl run` for executing presets.
* Implement cache modes (isolated/host/volume) with warnings.
* **TODO**: Push pre-built Docker images to Docker Hub public registry (enables 2-min pull vs 10-30 min build).

**90 days - Comparison & Automation**

* Build `cvl compare` (HTML/Markdown reports).
* Turn on weekly smoke CI for survivors only.
* Publish auto-generated docs gallery (MkDocs).
* Add stability badges (last-green date, GPU tier).

---

## 15) Standardization Audit (Oct 2025)

**Current State (59 dockerized examples):**
- ✅ 67/69 have `example.yaml` (97% coverage)
- ✅ 59/59 have `build.sh`
- ✅ 59/59 have `Dockerfile`
- ✅ 57/57 have `build` preset in example.yaml (100% coverage for examples with example.yaml)
- ✅ 0 scripts use fragile `../../../` paths (all fixed)

**Standardization Pattern (established via moondream2):**
```yaml
# example.yaml
presets:
  build:
    script: build.sh
    description: Build the Docker image
  predict/generate/train:
    script: <name>.sh
    description: <semantic description>
```

```bash
# Shell scripts pattern
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
CACHE_DIR="$REPO_ROOT/data/container_cache"
```

**Rollout Plan:**
1. **Phase 1 (Completed):** Establish pattern via moondream2 ✅
2. **Phase 2 (Completed):** Add build presets to all 56 examples ✅
3. **Phase 3 (Completed):** Fix fragile paths in 13 scripts ✅
4. **Phase 4 (Pending):** Document pattern in CONTRIBUTING.md

**Priority Examples for Standardization:**
- Stable perception: granite_docling, surya, moondream3
- Stable generative: minisora, wan2gp, nanochat
- High-traffic: flux, dreambooth, nanogpt

---

## 13) KPIs - REVISED

* Median **time‑to‑first‑experiment** (target: <5 min).
* % examples with **green weekly smoke** (target: >90%).
* **Copy‑outs** per month (`export` telemetry or template clones).
* **Return users** running ≥2 different examples/month.
* **% of examples removed per quarter** (target: 20-30% in first year) - health metric.
* **# examples graduated per quarter** (target: 2-3) - ecosystem growth metric.

---

## 14) Bottom Line

Keep CVlization laser‑focused on **fast runs, fair comparisons, clean copy‑outs**. Let examples carry the ML specifics; let a **tiny core** provide the protocols and tooling that make the repo feel cohesive, trustworthy, and immediately useful to busy engineers.

**Critical mindset shift:** The repo's health is measured by **what you remove**, not what you manage. Build export tooling first, archive aggressively, and maintain only 10-15 pristine examples. Successful examples should graduate to standalone repos—that's a win, not a loss.

