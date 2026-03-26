# OMR Project — Current Status

_Last updated: 2026-02-26_

---

## Implemented examples

| Example | Docker image | Status | Output |
|---------|-------------|--------|--------|
| `audiveris` | `cvlization/audiveris:latest` | ✅ built & verified | MusicXML `.mxl` |
| `smt_omr` | `cvlization/smt-omr:latest` | ✅ built & verified | ekern notation |
| `qwen3_vl` | `cvlization/qwen3-omr:latest` | ✅ files created | structured Q&A text |
| `vlm_omr` | `cvlization/vlm-omr:latest` | ✅ built & verified | JSON with probe responses |
| `omr_layout_analysis` | `cvlization/omr-layout-analysis:latest` | ✅ built & verified | JSON bounding boxes |
| `lilypond` | `cvlization/lilypond:latest` | ✅ built & verified | PNG/PDF/SVG from kern/MusicXML/LilyPond |

---

## Model comparison on vintage_score_1884.jpg (Biddle's Piano Waltz, 1884)

### Visual ranking (LilyPond renders)

**Gemini 3.1 Pro > Claude Opus 4.6 (w/ thinking) > SMT**

| Model | Key | Measures | Section labels | Notable issues |
|-------|-----|----------|---------------|----------------|
| Gemini 3.1 Pro | G major ✓ | ~41 | INTRO./WALTZ. ✓ | Intro pitches wrong (simplified chords vs ornaments); structure good |
| Claude Opus 4.6 (5k thinking) | C major ✗ | 38 | Intro/Waltz ✓ | Wrong key — cluttered accidentals; thinking doubled coverage (14→38 measures) |
| Claude Opus 4.6 (no thinking) | C major ✗ | 14 | Intro/Waltz ✓ | Stopped early, degenerate bass pattern |
| SMT (`PRAIG/smt-fp-grandstaff`) | B♭ major ✗ | ~41 | None | Note collisions, missing staves, structurally broken on vintage scan |
| GPT-5.2 Pro | C major ✗ | ~0 | None | Produced only rests |
| Audiveris | Failed | sparse | None | Spurious key changes, wrong beats |

### Key observations
- Gemini 3.1 Pro is the best model for this task on vintage scans
- Claude Opus 4.6: thinking budget (5k tokens) significantly helps coverage but doesn't fix key signature error
- SMT: trained on clean grandstaff crops — degrades badly on full-page vintage scans with noise
- All models struggle with the INTRO ornamental figures (chromatic runs, grace notes)

### SMT CER on clean GrandStaff sample
- After `_normalize_bekern()` fix (strip `@` and `·`): **CER 4.05%, SER 4.69%**
- Before fix: CER 26.36%, SER 43.69%
- Only 3 genuine musical errors in 228 lines (per sequence-aligned diff)

---

## Rendering pipeline (ekern → PNG)

Two backends available in `vlm_omr/omr_comparison/render_ekern.py`:

### Verovio (default)
- `--renderer verovio` (default)
- Deps: `verovio==6.0.1`, `cairosvg==2.8.2`
- Supports: title/composer via `"header": "encoded"`, dynamics `!!LO:DY`
- Limitation: `!!LO:TX` section labels silently dropped

### LilyPond (recommended)
- `--renderer lilypond`
- Uses `cvlization/lilypond:latest` Docker image (no host install needed)
- Output: `<model>_rendered_ly.png`
- Emmentaler font — better vintage look than verovio's Leipzig
- Supports: title, composer, tempo (`!!!OMD:`), section labels (`*>Label`) via `_inject_kern_metadata()`
- Fix applied: removes `\include "lilypond-book-preamble.ly"` (was causing title-only first page)
- Compact layout: `#(set-global-staff-size 16)` keeps content on one page

### Prompt (vlm_omr/predict.py — ekern_transcription probe)
- Music21-native kern for LilyPond pipeline:
  - `!!!OMD:` for tempo → `\tempo` in LilyPond
  - `*>Label TAB *>Label` for section labels → `\mark` in LilyPond
  - `!!!OTL:`, `!!!COM:` for title/composer
- Key change instruction: emit new `*k[]` record mid-piece
- `--thinking-budget N` flag added for Claude extended thinking
- Concrete example uses neutral content (not waltz-specific)

---

## OMR dataset landscape

OMR decomposes into four stages. Available datasets cover only the early stages:

```
OMR pipeline
├── Layout analysis        → omr_layout_analysis example (bounding boxes)
├── Symbol detection       → DeepScoresV2 (255k images, 136 classes, bbox only)
├── Notation reconstruction  (no public dataset; requires grouping symbols into notes/measures)
└── Score transcription    → output: ekern / MusicXML / MEI
```

### Symbol detection datasets (published on HuggingFace)

| Dataset | Images | Annotations | Labels | Hub |
|---------|--------|-------------|--------|-----|
| DeepScoresV2 dense | 1,714 | ~1.1M | 136 symbol classes | `zzsi/deep-scores-v2-dense` |
| DeepScoresV2 complete | 255,385 | ~120M | 136 symbol classes | `zzsi/deep-scores-v2` |

**Key limitation:** DeepScoresV2 provides bounding boxes for individual symbols but **cannot be used to reconstruct score transcriptions** (ekern/MusicXML) directly because:
- No barline annotations (measure boundaries unknown)
- No explicit links between related symbols (notehead ↔ stem ↔ flag ↔ dot)
- Pitch requires staff geometry inference from y-coordinates

### Score transcription datasets (synthetic rendered pairs)

| Dataset | Pages | Splits | Labels | Hub |
|---------|-------|--------|--------|-----|
| OpenScore (lieder + quartets + orchestra) | 6,360 | train/dev/test | MusicXML (full score) + rendered PNG | `zzsi/openscore` |
| OLiMPiC (synthetic) | ~17,945 | train/dev/test | Linearized MusicXML (LMX) per system | `zzsi/olimpic` |

`zzsi/openscore` was generated by the **LilyPond synthetic pipeline** (`datasets/omr/openscore/prepare.py`):
- Input: OpenScore MusicXML corpus (lieder 1,460 scores, quartets 122, orchestra 94)
- Rendering: MusicXML → music21 → LilyPond → PNG per page (via `cvlization/lilypond:latest`)
- Metadata injected: composer, opus, movement number/title visible on each rendered page
- Augment renders with vintage scan artifacts (noise, rotation, paper texture) to simulate degraded input

No public dataset provides paired **vintage scan image → ekern/MusicXML** examples. The synthetic pipeline above bridges this gap for pretraining; vintage robustness requires augmentation on top.

### HuggingFace multi-table design

HF datasets do not support relational tables natively (no joins/foreign keys). The closest equivalent is **multiple configurations** in one repo:

```python
dd_pages.push_to_hub("zzsi/openscore", config_name="pages")   # one row per page, has image
dd_scores.push_to_hub("zzsi/openscore", config_name="scores")  # one row per score, has musicxml
# Load: load_dataset("zzsi/openscore", "scores")
```

**Completed:**
- `zzsi/openscore-sft` — flat rows, single-page only, `(image, musicxml)` — ✅ pushed (326 train / 10 test / 16 dev)
- `zzsi/openscore` config `scores` — one row per score, full MusicXML, no image — ✅ pushed
- Multi-page SVG pipeline — ✅ implemented in `datasets/omr/openscore/page_musicxml.py`

**Next (not yet run at scale):** run `page_musicxml.py --corpus lieder --push-to-hub zzsi/openscore` to add per-page MusicXML to all 17k pages

**GT format:** Store **MusicXML** in the dataset (lossless, source format). Convert to LMX or bekern at training time if sequence length is a concern.

### SFT dataset suitability

| Dataset | Suitable for SFT? | Reason |
|---------|-------------------|--------|
| `zzsi/olimpic` | ✅ directly | System-level crops + LMX labels, ready to use |
| `antoniorv6/grandstaff` | ✅ directly | System-level crops + bekern labels |
| `zzsi/openscore` | ⚠️ partial | Images exist; per-page MusicXML labels need work (see below) |
| `zzsi/deep-scores-v2*` | ❌ | Bounding boxes only — no notation sequence labels, source MusicXML not released |

### Per-page label extraction for zzsi/openscore

The dataset has `n_pages` per score. Single-page scores are ground truth by construction (full MusicXML = page label):

| Corpus | Single-page scores | Total scores |
|--------|--------------------|--------------|
| lieder | 326 | 1,226 |
| quartets | 0 | 114 |
| orchestra | 1 | 84 |
| **Total** | **327** | **1,424** |

For multi-page scores, per-page label extraction using the **SVG bar-number approach** — ✅ implemented:

**How it works (`datasets/omr/openscore/page_musicxml.py`):**
1. Render MXL → SVG (one file per page) via Docker with `\set Score.barNumberVisibility = #all-bar-numbers-visible` — makes ALL bar numbers visible in every system (including broken-bar continuations at page breaks)
2. Parse SVG `<text>` elements: bar numbers are identified by the most-common font size; when tied (sparse page) prefer the size whose candidates have the highest value (bar numbers > time-sig denominators); filter outliers by keeping the largest consecutive-integer cluster
3. Page N: `bar_start = min(bar_nums_on_page_N)`, `bar_end = min(bar_nums_on_page_N+1) - 1`
4. Pickup-bar offset: if `score.parts[0].measures[0].number == 0`, subtract 1 from SVG bar numbers before calling `music21.score.measures()`
5. Slice MusicXML: `music21.score.measures(bar_start, bar_end)` → per-page MusicXML
6. SVGs cached in `~/.cache/openscores/svg/{corpus}/` — batch render via single Docker call

**Validated on (sliced=total, gap=0):**
- Mahler Kindertotenlieder No.1 (2 pages, 85 measures, pickup bar, offset=1) ✅
- Mahler Rheinlegendchen (13 pages, 120 measures, offset=0) ✅ — fixed time-sig contamination on sparse page 9

**Known limitation:** measures that span page boundaries (broken bars) are assigned to the NEXT page. Acceptable for SFT training.

### LilyPond Scheme research (archived)

Scheme-based page layout queries are not viable: `after-line-breaking` is a dummy property, page numbers unavailable at Scheme level, C++ required. The SVG bar-number approach above is the correct solution.

---

## VLM SFT — MXC experiments (2026-03-14 to 2026-03-24)

### Output format: MXC (MusicXML Compact)

Raw MusicXML is too verbose for VLM training — a typical page is ~14K tokens,
far exceeding the 4096-token context window. No model ever generated actual
`<note>` elements within the inference token budget.

We developed **MXC** (`vlm_omr_sft/mxc.py`), a line-based compact encoding
achieving 12x compression with 99.2% lossless round-trip accuracy on 500 samples.
Median page: ~1,200 MXC tokens. See `vlm_omr_sft/MXC.md` for format spec.

### Model comparison (all MXC, r=32, 3K lieder, best checkpoint)

| Model | Size | Pitched-only sim | Rhythm | eval_loss |
|---|---|---|---|---|
| DeepSeek-OCR-2 | 3B | 0% | 4% | 0.315 |
| Gemma-3 4B | 4.4B | 1% | 11% | 0.216 |
| Ministral-3 | 3.9B | ~0% | 10% | 0.109 |
| Qwen3-VL 8B | 8B | 16% | 23% | 0.166 |
| Qwen3-VL 32B | 32B | 23% | 31% | 0.147 |
| **Qwen3.5-9B r=32** | **9.5B** | **35%** | **46%** | **0.149** |

**Pitched-only similarity**: alignment-aware sequence match on actual notes
(excludes rests). Measured by `eval_mxc.py` on 10 held-out samples.

### Key findings

1. **MXC format is essential** — without it, no model reaches note content.
2. **Qwen3.5-9B is the best model** — beats Qwen3-VL 32B (3.5× bigger) thanks
   to architecture improvements (unified early fusion in Qwen3.5 vs Qwen3-VL).
3. **Models < 8B cannot learn pitch** regardless of architecture or LoRA rank.
4. **LoRA r=32 is optimal** for Qwen3.5-9B (tested r=8, 16, 32, 64).
5. **2-3 epochs is optimal** on 3K lieder samples. All models overfit after.
6. **Model has converged at 35% pitched-only similarity** — bottleneck is now
   training data, not model capacity or training duration.
7. **Best-sample accuracy is high**: 88% positional pitch accuracy, 83% sequence
   similarity, 123 consecutive matching events — the model CAN read pitch from
   images, just not consistently across all pages.

### What works vs what doesn't

**Works well:** MXC syntax, part structure, key signatures (80%), lyrics
(bilingual), pitch on simpler passages, rhythm/note types (46%).

**Struggles with:** dense piano polyphony, very long pages, pages with many
rest measures before note entry.

### Detailed reports

- `vlm_omr_sft/reports/2026-03-14-lieder-2epoch.md` — Gemma-3 initial run
- `vlm_omr_sft/reports/2026-03-18-model-comparison-and-data-cleaning.md`
- `vlm_omr_sft/reports/2026-03-19-mxc-experiments.md` — all MXC experiments
- `vlm_omr_sft/reports/evaluation-guide.md` — how to evaluate results

---

## Pending work

### Immediate (diagnostic)
- [ ] Synthetic data difficulty ladder — generate simple→complex pages to
  isolate where pitch discrimination fails (see plan.md for details)
- [ ] Full dev set evaluation (193 samples) for statistically meaningful accuracy

### Medium term
- [ ] RL with musical correctness reward (valid MXC, measure durations, pitch range)
- [ ] Segmentation pipeline: omr-layout-analysis → SMT system-by-system
- [ ] Vintage scan robustness (augmentation, domain adaptation)

### Open questions
- At what page complexity does pitch accuracy break down? (→ synthetic data)
- Would RL improve beyond SFT ceiling, or is more supervised data needed first?
- Does the model actually condition on the image, or memorize corpus statistics?
