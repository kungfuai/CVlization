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
* **CLI** (`cvl` via `pipx`):
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
cvl run  nlp/bert --preset train --cache host
cvl run  vis/resnet --preset eval --profile gpu
cvl compare runs/a.json runs/b.json --key val/accuracy
cvl export  vis/resnet --as-repo  # minimal standalone repo w/ pins + CI
cvl doctor                    # docker/nvidia/cache/disk checks
```

**Presets** provide contrastive comparisons (e.g., baseline vs policy aug vs mixing; LoRA vs full fine‑tune; CPU vs GPU).

---

## 5) Caching & Data Reuse

**Default:** Repo-scoped cache (`./data/container_cache` → container `/root/.cache/huggingface`) for isolation & reproducibility.

**Opt-in modes:**
* `--cache=host`: Mount `~/.cache` to reuse models downloaded outside Docker (saves disk space, faster cold starts)
  * ⚠️ Warning displayed about non-determinism
* `--cache=volume`: Named Docker volume (macOS performance optimization)

Respect official envs: `HF_HOME=/root/.cache/huggingface`, `TORCH_HOME=/root/.cache/torch`.

---

## 6) Trust & Reproducibility

* **Scheduled CI smoke builds** (weekly): build images + 30‑sec CPU check per example.
* **Badges**: last‑green date, GPU required/optional, stability tier (Stable/Beta/Experimental).
* **Run‑record JSON** written on each run → source of truth for `cvl compare`.

---

## 7) Docs as a Gallery

* Auto‑generate a searchable gallery (MkDocs) from `example.yaml` + READMEs.
* Filters: task, modality, GPU/VRAM, ETA, stability; quick copy buttons for `cvl` commands.

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
3. Won't force core to pin heavy ML libs?
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

## 12) Roadmap - UPDATED (Oct 2025)

**Completed ✅**
- ✅ Add `example.yaml` to 97% of examples
- ✅ Build `cvl list` and `cvl info` with fuzzy matching
- ✅ Build `cvl run` for executing presets
- ✅ Standardize 160 scripts across all examples
- ✅ Add directory mounting info to `cvl run` output
- ✅ Fix all fragile hardcoded paths

**Next 30 days**
- Push pre-built Docker images to Docker Hub (2-min pull vs 10-30 min build)
- Build `cvl export --as-repo` (standalone repo with pins + CI)
- Audit examples: tag Green/Yellow/Red, archive Red ones
- Graduate 3-5 high-maintenance examples (>500MB artifacts)

**Next 60 days**
- Add run-record JSON schema and writer
- Build `cvl compare` (HTML/Markdown reports)
- Turn on weekly smoke CI for stable examples
- Publish auto-generated docs gallery (MkDocs)

**Next 90 days**
- Add stability badges (last-green date, GPU tier)
- Identify 10-15 Green examples for long-term maintenance
- Move `cvlization/torch` and `cvlization/tensorflow` to `/legacy`

---

## 13) KPIs

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

---

## 15) Standardization Status (Oct 2025) ✅ COMPLETE

**Scripts Standardized (160 total across 5 commits):**
- ✅ 48 build.sh files → use `$SCRIPT_DIR` pattern
- ✅ 29 train.sh files → standardized CVL pattern
- ✅ 13 predict.sh files → standardized CVL pattern
- ✅ 23 other scripts → GPU flag updates, test scripts
- ✅ 67/69 have `example.yaml` (97% coverage)
- ✅ 0 scripts use `--runtime nvidia` (all use `--gpus=all`)
- ✅ 0 scripts use fragile hardcoded paths

**Standard Pattern:**
```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"  # Go up 4 levels
IMG="${CVL_IMAGE:-example_name}"

docker run --rm --gpus=all --shm-size 16G \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
  --env "PYTHONPATH=/cvlization_repo" \
  "$IMG" python train.py "$@"
```

---

## 16) Examples Requiring Manual Downloads

**12 examples need manual setup before running:**

### Video Generation (6 examples - run download_models.sh):
- `animate_x` - Animate-X + DWPose models (~2-3GB from HF)
- `mimic_motion` - MimicMotion + DWPose models (HF)
- `phantom` - Phantom model weights
- `vace` - VACE model weights
- `vace_comfy` - Wan 2.1 models (~15-20GB from HF)
- `wan_comfy` - Wan 2.1 models (similar to vace_comfy)

### Tracking (2 examples - use gdown):
- `global_tracking_transformer` - Soccer video + GTR model from Google Drive
- `soccer_visual_tracking` - `bash download_data.sh` (video + 3 YOLO models)

### LLM (3 examples - HF token/access):
- `mistral7b` - Gated model, requires HF account + access request
- `mixtral8x7b` - Gated model, requires HF account + access request
- `trl_sft` - Llama models, requires `export HF_TOKEN=...`

### Other (1 example):
- `nanochat` - `git clone https://github.com/karpathy/nanochat $HOME/zz/nanochat`

**All other examples** auto-download models/data on first run (HuggingFace, datasets, etc.)

---

## 17) CLI UX Improvements

**Completed ✅**
- ✅ Show "Running..." header with example/script/docker info
- ✅ Show directory mounting information
- ✅ Fuzzy matching with suggestions for typos
- ✅ Check Docker is running before execution

**TODO (Ship when users ask):**
- Handle Ctrl+C gracefully (no ugly tracebacks)
- Better missing image error + auto-build prompt
- Show completion time (✓ Completed in 5m 32s)
- VRAM/disk space checks (⚠️ fragile, wait for user requests)

**Philosophy:** Keep it simple. Don't try to be smarter than bash, just friendlier.
