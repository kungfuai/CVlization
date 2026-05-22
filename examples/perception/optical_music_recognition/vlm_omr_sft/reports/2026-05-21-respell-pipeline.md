# L7a Plateau Broken: Deterministic Re-Spell Pipeline (2026-05-21)

**Previous**: [`2026-05-16-qwen3vl-key-signature.md`](2026-05-16-qwen3vl-key-signature.md)
— diagnosed the L7a "plateau" as a wrong-key belief that cascades through
note spellings; concluded RL, hints, aux mixing, capacity, and key-in-prompt
all fail to dislodge it.

## TL;DR

- The L7a plateau (mean pitched ~70%) was almost entirely a key cascade.
  Decoupling the metric showed **position skill is already at ceiling
  (96.9% mean)**.
- A **deterministic re-spell at inference time** — given a trusted key,
  recompute each non-explicit-accidental note's alter — recovers the
  cascade mechanically.
- **End-to-end with a perfect Stage-1 key (GT)**: pitched mean **57.2% →
  97.1%**, median **39.1% → 99.2%**. Every previously-broken intermediate
  key recovered to ~95-97%. The mechanism works.
- The remaining gap is Stage 1: a focused image→key classifier converged
  to a wrong-by-one mapping on +3 and −4 (54% per-key). A crop+zoom probe
  ruled out resolution → the off-by-one is *representational*, not
  perceptual. CoT target + wider data (v2) is the current attempt.
- MXC3 (decoupled per-aspect format) was built to 84% openscore round-trip
  but **deprecated** in favor of this simpler pipeline.

## Pipeline overview

```
                ┌─────────────────────┐
   page image ──┤ Stage 1: image→key  │── key=N
                └─────────────────────┘
                ┌─────────────────────┐
   page image ──┤ Stage 2: image→MXC2 │── transcription (with possibly wrong key cascade)
                └─────────────────────┘
                              │
                              ▼
                ┌─────────────────────┐
        key=N + transcription ─→ ┤ respell.py: replace  │── corrected transcription
                ┤ key-implied alters    │
                └─────────────────────┘
```

- **Stage 1**: focused `image → key=N` classifier (separate model or head).
- **Stage 2**: standard full-page MXC2 transcription model (e.g.,
  `safckylj`). Note spellings may be wrong due to wrong-key belief.
- **Re-spell**: deterministic. For each note with no explicit `acc=`
  marker in Stage 2's output, recompute its accidental using Stage 1's
  key. Notes with `acc=` are explicit chromatic deviations and are
  preserved.

## Why this works mechanically

The Level 7a investigation showed:
- The model gets *note positions* essentially correct on broken samples
  (~85-95%).
- The metric collapses because absolute pitch bundles position + key.
- A wrong key sharps every D on the page → 38 wrong pitch tokens from one
  decision.

Re-spell separates these concerns at decode time. Notes with explicit
accidentals (the `acc=` marker indicates the engraving showed a glyph)
are preserved verbatim. Notes without explicit accidentals had their
alter derived from the model's perceived key — and that's exactly what
gets corrected by re-spelling with the trusted Stage-1 key.

## E2E validation with GT key (Stage-1 oracle)

Run: `safckylj` + GT key from dataset + `respell.py`, L7a dev (n=50):

| Metric | base | respell |
|---|---|---|
| pitched mean | 57.2% | **97.1%** |
| pitched median | 39.1% | **99.2%** |
| position mean | 97.1% | 97.1% |

Per-key — the previously-broken intermediates recovered to ceiling:

| key | n | base | **respell** | gain |
|---|---|---|---|---|
| +1 | 5 | 13.6% | **95.2%** | +81.5pp |
| +3 | 10 | 29.8% | **96.7%** | +66.9pp |
| +2 | 7 | 31.0% | **97.2%** | +66.3pp |
| −1 | 5 | 18.7% | **94.4%** | +75.7pp |
| −3 | 1 | 25.6% | **96.5%** | +70.9pp |
| 0, ±4, −2 | (already-good) | ~98% | ~98% | unchanged |

This is the **ceiling**: what the pipeline achieves if Stage 1 is perfect.

## Stage 1 attempts so far

### v1 — single-task `key=N`, L7a only (n=4000, 2 epochs)

- Eval on L7a dev (n=50): **27/50 = 54%** overall.
- Systematic off-by-one error:
  - +3 → +4 (0/10), +2 → +3 (6/7 wrong), −4 → −3 (7/7 wrong)
  - All other keys: 100% correct (-3, -2, -1, 0, +1, +4)
- Best eval_loss at step 200; cp600 ≈ 56%; later checkpoints didn't help.

### Crop+zoom probe (rules out resolution)

Cropped top-30%×55% of page (covers all 3 staves' key signatures),
upscaled 3× → fed to v1 classifier. Same 54% with same off-by-one
pattern. **Resolution is not the bottleneck** — the encoder sees the
glyphs fine. The off-by-one is representational/learned.

### v2 — CoT target + wider data (in progress)

Hypothesis: the simple `key=N` target collapses +3 and +4 (one-token
difference), so the model learns one mapping that minimizes loss across
both. A chain-of-thought target makes them differ by multiple tokens
(content + count word + key=N), giving stronger discriminative gradient.

- Target format:
  - `+3` → `"Sharps: F C G. Three sharps. key=3"`
  - `+4` → `"Sharps: F C G D. Four sharps. key=4"`
- Training data: L5 + L6 + L7a + L8 + L9 combined (~12K samples) for
  rendering / content variety.
- Status: OOM-killed at step 1035 of 5936. Checkpoint-600 saved; eval
  pending.

## What's deprecated

MXC3 — the per-aspect decoupled format that aimed to make the VLM's job
easier — is no longer the active path. It reached 100% round-trip on
synthetic L7a/L9 but only 84% on openscore lieder, and the decoupling
motivation (untested speculation that channels would help training) is
not needed once the cascade is handled by re-spell. See `MXC3.md` header
and the deprecated `mxc3*.py` files.

## Files

| File | Purpose |
|---|---|
| `mxc2_slice.py` | Stateless per-measure MXC2 slicer |
| `respell.py` | Deterministic re-spell of non-explicit accidentals |
| `eval_respell.py` | E2E pipeline eval (GT or classifier key) |
| `eval_keyclassifier.py` | Stage 1 per-key accuracy |
| `eval_keyclassifier_crop.py` | Resolution probe (crop+zoom) |
| `config_qwen3vl_keyclassifier.yaml` | Stage 1 v1 (plain target) |
| `config_qwen3vl_keyclassifier_v2.yaml` | Stage 1 v2 (CoT + wider data) |
| `config_qwen3vl_l7a_per_measure.yaml` | Stage 2 per-measure (planned) |
| (deprecated) `mxc3.py`, `mxc3_decode.py`, `mxc3_slice.py`, `MXC3.md` | per-aspect format, retained for reference |

## Current open question

Can a focused Stage-1 classifier reach ~99% per-key — specifically on the
+3 / +4 / −3 / −4 boundary where the encoder shows a learned off-by-one?

- If yes → pipeline hits the ~97% ceiling end-to-end. Plateau broken.
- If no → an alternate Stage 1 (small CNN, structured binary heads,
  Audiveris-symbol fallback on the key-sig region) becomes necessary.

The v2 run will give the first data point on this.
