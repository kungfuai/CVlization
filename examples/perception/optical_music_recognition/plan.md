Vintage Sheet Music Digitization (LLM/VLM/OMR)
What we’re building

Goal: take an image/PDF scan of a (often vintage) piano score and produce outputs that are useful and credible.
Key tension:

* Facsimile fidelity (colors, fonts, decorative cover pages, print imperfections, handwritten markings)
* Musical semantics (notes, rests, voices, lyrics, dynamics, structure; editable + playable)

Recommended product framing (so you don’t force a false tradeoff):

* Dual output:

    1. Searchable facsimile: original scan preserved as the “truth layer” + overlays
    2. Symbolic transcription: MusicXML +/or MEI (and optionally LilyPond)

* Alignment between them (where possible): connect semantic events to image regions (zones).

Relevant standards/tools:

* MEI (Music Encoding Initiative): https://music-encoding.org/
* LilyPond (text-based engraving): https://lilypond.org/
* MuseScore (editing MusicXML and more): https://musescore.org/


Data: what exists and what’s realistically “vintage messy”

You explicitly asked: large, truly vintage, degraded paper, ornate fonts/colors, no ground-truth required.

A) Large downloadable vintage scan sources (no labels)

These are best thought of as digital library corpora rather than ML datasets.

* Gallica / BnF (France): IIIF access for images (good for programmatic harvesting):
    * IIIF API overview: https://api.bnf.fr/fr/api-iiif-de-recuperation-des-images-de-gallica
    * Gallica portal: https://gallica.bnf.fr/
* Library of Congress (US): large sheet music collections + APIs:
    * Example collection (1870–1885): https://www.loc.gov/collections/american-sheet-music-1870-to-1885/
    * LoC APIs: https://www.loc.gov/apis/
* NYPL Digital Collections (US):
    * Public-domain sharing post + guidance: https://www.nypl.org/blog/2016/01/05/share-public-domain-collections
    * NYPL API (for programmatic access): https://api.repo.nypl.org/
* Europeana (EU aggregator): https://apis.europeana.eu/
* Internet Archive: https://archive.org/
* IMSLP (huge scan corpus; download is usually per-item/edition; rights vary by region): https://imslp.org/
* Sheet Music Consortium (discovery/metadata hub pointing to partner libraries): https://digital.library.ucla.edu/sheetmusic/

What these give you:

* Real paper texture, ink fading, marginalia, publisher typography, ornate covers, color lithography, warping, scan shadows.

What they don’t give you:

* A uniform schema, easy bulk packaging, or ground-truth musical encodings.

B) Open OMR datasets (labeled) — mostly NOT truly vintage

These are valuable for pretraining/bootstrapping, but you shouldn’t pretend they solve “vintage.”

* DeepScoresV2 (very large, mostly synthetic/rendered): https://zenodo.org/records/4012193
* DoReMi (born-digital prints + rich metadata; paper): https://arxiv.org/abs/2107.07786
* PrIMuS / Camera-PrIMuS (incipits; Camera variant simulates photo artifacts): https://grfia.dlsi.ua.es/primus/
* MUSCIMA++ / CVC-MUSCIMA (handwritten music; different domain than printed vintage): https://ufal.mff.cuni.cz/muscima
* OLiMPiC (pianoform systems; includes formats for end-to-end training; repo): https://github.com/ufal/olimpic-icdar24

Common pattern:

* Many large labeled datasets are synthetic/born-digital renders (excellent for learning music structure, poor for real scan artifacts).
* Some datasets simulate camera artifacts (good for skew/blur, still not antique engraving).
* A few include real scanned splits.

Practical takeaway:

* Use labeled OMR datasets to learn semantics.
* Use large library scans to learn robustness + style + layout variance (with weak/self-supervision).


Methods: system architecture that can ship + improve

Core pipeline (MVP → production)

1. Ingest: image or PDF (extract pages)
2. Preprocess:

    * deskew + crop
    * dewarp (for photos)
    * denoise + contrast normalization (keep color channel for “fidelity” work)

1. Detect & split: staff lines / systems / measures (piano grand staff)
2. Transcribe semantics: image → token sequence → MusicXML/MEI
3. Postprocess constraints:

    * meter consistency
    * voice/beam sanity
    * measure durations sum correctly

1. Alignment layer (optional at first, then core differentiator):

    * map predicted events back to image regions
    * emit MEI facsimile zones and link them to events

1. Outputs:

    * Original scan PDF (archival)
    * MusicXML (editing in MuseScore etc.)
    * MEI (for facsimile linkage and research workflows)
    * Optional clean re-render (SVG/PDF)

Rendering/visualization options:

* Verovio (render MEI to SVG, used widely in DH/musicology): https://github.com/rism-digital/verovio
* mei-friend (MEI editor/viewer; helpful for facsimile workflows): https://github.com/mei-friend/mei-friend

Representation choice for training

Training directly to raw MusicXML is often brittle.
Recommended:

* Train to a linearized encoding (e.g., “LMX-like” sequences or a compact grammar) and convert to MusicXML/MEI.
* Keep a strict grammar to reduce hallucinated XML and make evaluation easier.

Model “buckets” to compare

You want multiple models + techniques for an experiments story.
Bucket 1 — baseline you can demo immediately:

* Audiveris (open-source OMR engine; exports MusicXML): https://audiveris.github.io/audiveris/

Bucket 2 — OMR-native end-to-end transformers:

* SMT (Sheet Music Transformer) repo: https://github.com/antoniorv6/SMT
* SMT paper (image-to-seq for complex scores): https://arxiv.org/abs/2402.07596

Bucket 3 — general vision transformers (fine-tuned small vs. prompted large):

Key question: does fine-tuning a small model beat prompting a large one for OMR?

* Fine-tuned small models:
    * Donut (OCR-free doc understanding): https://arxiv.org/abs/2111.15664
    * Pix2Struct (screenshot parsing pretraining): https://arxiv.org/abs/2210.03347
* Zero-shot / few-shot prompted large VLMs:
    * Qwen2.5-VL (strong on documents and grounding/bbox output): https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
    * LLaVA-OneVision: https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct
* Use as correction assistant, consistency checker, or interactive UX layer regardless of which wins on raw OMR accuracy

Bucket 4 — facsimile & restoration:

Three distinct sub-tasks (can be tackled independently):

1. Document restoration: deskew, dewarp, denoise, debind shadow — improve scan quality while preserving print style
    * DocTr (geometric correction): https://github.com/mindee/doctr
    * DiffBIR (diffusion-based blind image restoration): https://github.com/XPixelGroup/DiffBIR

2. Zone/layout detection: locate systems, measures, and symbols in the image → emit MEI `<facsimile>` zone coordinates that link semantic events back to image regions
    * Any object detection backbone (DINO, RT-DETR) fine-tuned on annotated score layouts
    * Rendering output: Verovio (MEI → SVG with zone links): https://github.com/rism-digital/verovio

3. Style-preserving re-engraving: from symbolic output (MusicXML/MEI), generate a clean notation image rendered in the original publisher's visual style (typography, spacing, ornamental elements)
    * FLUX.1-dev + ControlNet (open weights; better structural fidelity than SD 1.5 for fine notation detail): https://huggingface.co/black-forest-labs/FLUX.1-dev
    * ControlNet + Stable Diffusion with a vintage LoRA (train LoRA on IMSLP-sourced scans grouped by era/printing technique)
    * CycleGAN for unpaired domain transfer (clean engraving ↔ degraded vintage scan)
    * Reference: style_transfer_idea.md in this directory


Training strategy: how to make progress fast on “vintage”

The hard truth

Large, fully-labeled vintage data is rare.
So you win by:

1. Pretraining on abundant labeled (mostly synthetic) OMR corpora
2. Domain-adapting on unlabeled vintage scans using weak/self-supervision + pseudo-labeling
3. Spending a small amount of human time where it matters (active learning)

Suggested 3-stage plan

Stage 1 — Semantic transcription MVP (fastest path to a demo)
Objective: produce editable MusicXML with acceptable accuracy on moderately clean scans.
Training data:

* Labeled OMR datasets (synthetic + any scanned subsets)
* System-level crops for pianoform to reduce complexity

Models:

* OMR-native transformer (primary)
* Document-parsing transformer (secondary)
* Classical OMR engine baseline

Outputs:

* MusicXML
* Playback + error highlighting

Success criteria:

* “Works on 60–80% of pages with light correction”

Stage 2 — Vintage domain adaptation (the real product)
Objective: robustness to aged paper, ornate fonts, weak contrast, stains, bleed-through.
Data:

* 10k–200k unlabeled pages from Gallica/LoC/NYPL/IA (choose a narrow target style first)

Techniques:

* Strong augmentations on labeled data: paper textures, fading, blur, skew, bleed-through
* Pseudo-labeling:
    * run baseline OMR on vintage
    * keep high-confidence regions as training targets
    * ignore uncertain regions
* Active learning:
    * sample pages where model is most uncertain
    * pay for minimal human corrections

Success criteria:

* “Works on your target vintage style reliably; confidence highlights what needs review.”

Stage 3 — Facsimile-linked edition (differentiator)
Objective: don’t just transcribe; produce a scholarly/archival digital edition.
Techniques:

* Detect/track zones (systems, measures, symbols)
* Link zones to semantics
* In viewer: original scan + hover/click highlights + side-by-side clean engraving

Success criteria:

* “Preservation-grade + searchable + explainable.”


Experiments to run (a concrete matrix)

Here is a set of experiments that produces publishable graphs and a strong demo.

Experiment group A — Output format ablation

* A1: raw MusicXML tokens
* A2: linearized encoding (LMX-like)
* A3: constrained grammar decoding (CFG / token masks)

Metrics:

* syntax validity rate
* conversion success rate
* semantic accuracy (SER / OMR-NED-like)

Reference (example venue write-up for SER/OMR-NED style evaluation): https://ismir2025program.ismir.net/poster_70.html
Expected outcome:

* A2/A3 dramatically improve robustness and reduce broken outputs.

Experiment group B — Model family comparison

* B1: OMR-native transformer (SMT)
* B2: general vision transformer — fine-tuned small (Donut/Pix2Struct) vs. prompted large (Qwen2.5-VL)
* B3: baseline engine (Audiveris)

Metrics:

* semantic accuracy on clean test
* degradation robustness on vintage test
* correction time (human minutes per page)

Expected outcome:

* baseline engine strong on clean scans; OMR transformer best after fine-tune; doc model competitive if decoding constrained.

Experiment group C — Vintage adaptation techniques

* C1: augmentations only
* C2: augmentations + pseudo-labels (high-confidence only)
* C3: C2 + active learning (human fix top-uncertainty)

Metrics:

* accuracy vs pages labeled
* uncertainty calibration (confidence vs correctness)
* “time to acceptable” per page

Expected outcome:

* C3 yields biggest real-world gains per dollar.

Experiment group D — Layout/style retention

Even if you don’t fully “re-engrave faithfully,” you can quantify stylistic elements.

* D1: keep facsimile only (perfect fidelity)
* D2: re-render clean engraving
* D3: hybrid: re-render but preserve text fonts/colors as extracted

Metrics:

* text style accuracy (font class, color, placement)
* layout similarity (staff spacing, measure widths)

Expected outcome:

* D1 wins fidelity; D3 is the “premium” version.

Experiment group E — Facsimile & restoration pipeline

* E1: restoration as preprocessing — does DocTr + DiffBIR denoising improve downstream OMR accuracy (SER/CER) on degraded vintage scans?
* E2: zone detection — measure IoU of predicted system/measure bounding boxes vs. manual annotations; then link to MEI facsimile zones
* E3: style-preserving re-engraving — ControlNet+LoRA conditioned on edge map extracted from MusicXML-rendered structure; evaluate with FID and LPIPS vs. held-out vintage pages from same era

Metrics:

* E1: SER/CER on vintage test set (with vs. without restoration)
* E2: zone IoU, MEI facsimile validity
* E3: FID, LPIPS, human preference score

Expected outcome:

* E1 gives free accuracy gains on degraded inputs; E3 is the “wow” demo differentiator for archival/publisher clients.


Estimated cost (honest + actionable)

Because GPU pricing varies wildly by provider and month, the clean way to estimate is in GPU-hours.

Data costs

* Vintage scan harvesting: mostly engineering time (API + IIIF pipeline)
* Storage: depends on resolution and volume
    * Rough mental model: 1 page image might be 1–10 MB (sometimes more)
    * 100k pages → ~0.1–1 TB

Labeling costs (optional, minimal but high leverage)

* Active learning corrections:
    * If you do system-level (not full page) corrections, humans move faster
    * Budget idea:
        * 200–1,000 pages corrected can be enough for strong adaptation

Training compute (orders of magnitude)

Assuming system-level crops (piano systems) and a medium model:
Stage 1 (semantic MVP):

* Fine-tune: ~200–2,000 GPU-hours (depends on model size, batch, resolution, dataset size)

Stage 2 (domain adaptation):

* Continued training with pseudo-labels: ~500–5,000 GPU-hours

Stage 3 (alignment / detection):

* Symbol/zone detector training: ~100–1,000 GPU-hours

Total prototype program:

* ~800 to ~8,000 GPU-hours is a realistic “serious but not insane” range.

Translate to dollars with your chosen GPU-hour price:

* Total $ ≈ (GPU-hours) × (your $/GPU-hour)

If you want a minimal impressive demo on a tight budget:

* Keep everything at system-crop scale
* Do Stage 1 + a small Stage 2 active-learning loop
* That can plausibly fit in hundreds to low-thousands of GPU-hours.


Demo plan (what to show clients)

A demo should prove 3 things: fidelity, editability, and trust.

“60-second wow” flow

1. Upload a vintage scan
2. UI shows:

    * Left: original scan
    * Right: clean re-render (optional)
    * Hover: highlight corresponding region on scan
    * Confidence heatmap: “these measures likely wrong”

1. Export:

    * Original PDF (unchanged)
    * MusicXML (editable)
    * MEI with facsimile linkage (premium)

Why this sells

* Archives/publishers care about preserving the artifact
* Musicians/educators care about playback + editing
* Everyone cares about knowing what’s uncertain


Podcast content: a compelling narrative + episode outline

Core story

“Music is a visual language. OCR learned to read letters; OMR has to read polyphony, 2D structure, and typography.”
The hook:

* A 100-year-old score is not just notes; it’s a designed object.

The twist:

* Don’t choose between fidelity and semantics — ship a facsimile-linked edition.

Episode outline (30–45 minutes)

1. Why OMR is harder than OCR

    * 2D layout, multiple simultaneous voices, tiny symbols change meaning

1. What ‘good’ looks like for vintage

    * The artifact matters (paper, ink, typography)

1. The engineering approach

    * system crops
    * constrained decoding
    * post-processing constraints

1. How you get vintage robustness without labels

    * augmentations
    * pseudo-labels
    * active learning

1. Demo + failures (important!)

    * show success cases
    * show what breaks and how confidence reveals it

1. Where this goes next

    * faithful style-preserving re-engraving
    * large-scale music search across historical corpora
    * accessibility (screen readers, learning tools)

Great soundbites

* “We’re not replacing the scan; we’re adding a layer of meaning on top of it.”
* “The model doesn’t just output a file — it outputs its own uncertainty.”
* “Vintage digitization is a domain adaptation problem disguised as OCR.”


Next steps (what I’d do first)

1. Choose a single target niche for the first 4 weeks:

    * e.g., 1890–1930 US sheet music, piano + lyrics, decorative covers

1. Harvest 5k–20k pages (unlabeled) from 1–2 sources (keep it narrow)
2. Assemble a small labeled seed set:

    * 200–500 systems/pages with corrections (active learning)

1. Train Stage 1 MVP + run Stage 2 adaptation loop
2. Build the demo UI around “searchable facsimile” with confidence and export

If you want, I can turn this into a week-by-week execution plan with specific milestones, model sizes, and a minimal architecture diagram for the demo app.

