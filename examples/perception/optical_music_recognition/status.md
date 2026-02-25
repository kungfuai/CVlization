# OMR Project — Current Status

_Last updated: 2026-02-25_

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

### Score transcription datasets (what we need)

No public dataset provides paired (vintage scan image → ekern/MusicXML) examples.
The path to building one is the **LilyPond synthetic pipeline**:
- Render clean scores from kern source → perfect GT for free
- Augment renders to simulate vintage scan artifacts (noise, rotation, paper texture, dilation/erosion)
- This gives a supervised training set for end-to-end score transcription

---

## Pending work

### Short term
- [ ] PIL sepia/aged-paper post-processing for vintage look
- [ ] Run GPT-4o/GPT-5 with updated prompt
- [ ] Investigate why Claude Opus 4.6 consistently reads key as C major despite prompt

### Medium term
- [ ] Segmentation pipeline: omr-layout-analysis → SMT system-by-system
- [ ] Ground truth: search IMSLP for clean edition of Biddle's Piano Waltz
- [ ] Verifier module: syntactic validity, beat counts, mixed-duration chord detection

### Open questions
- Can Claude Opus 4.6 key signature accuracy be fixed with more explicit prompt instruction?
- Can SMT system-level model improve key signature accuracy on vintage scans?
- Does verovio 7.x support `!!LO:TX` from humdrum?
