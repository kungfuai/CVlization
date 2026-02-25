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

Motivation: Buckets 1-2 cover semantic transcription reasonably well on moderately clean scans, but vintage material introduces gaps that OMR-specific models weren't designed for:

* Image quality: foxing, bleed-through, severe fading, non-uniform illumination, and ornate typography confuse rule-based staff detectors (Audiveris) and degrade sequence model accuracy (SMT)
* Unusual layouts: lyrics between staves, mixed text/music pages, decorative covers, non-standard ornamental notation
* Zero-shot flexibility: can prompt a large VLM to extract only tempo markings, identify the printing era, or describe a decorative cover — without any fine-tuning

Comparison matrix:

|                | Small model        | Large model                        |
|----------------|--------------------|------------------------------------|
| Zero-shot      | —                  | Qwen3-VL prompted (no training)    |
| SFT            | Donut fine-tuned   | Qwen3-VL-8B fine-tuned on OMR data |
| RL             | optional           | GRPO with musical correctness reward |

Models:

* Small (fine-tune candidates):
    * Donut (OCR-free doc understanding): https://arxiv.org/abs/2111.15664
    * Pix2Struct (screenshot parsing pretraining): https://arxiv.org/abs/2210.03347

* Large VLMs (zero-shot or fine-tune, as of early 2026):
    * Qwen3-VL (Oct 2025; 8B/32B; 256K context; 32-language OCR; grounding): https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
    * Qwen2.5-VL (Jan 2025; 7B/72B; strong DocVQA and bbox grounding): https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
    * DeepSeek-VL2 (Dec 2024; MoE 4.5B active; OCRBench 834, DocVQA 93.3%): https://huggingface.co/deepseek-ai/deepseek-vl2
    * InternVL3.5 (Aug 2025; 1B–38B; strong reasoning): https://huggingface.co/OpenGVLab/InternVL3.5-8B
    * Molmo 2 (2025; 8B; best-in-class grounding/bbox output — relevant for zone detection in Bucket 4): https://huggingface.co/allenai/Molmo-7B-D-0924

* Use as correction assistant, consistency checker, or interactive UX layer regardless of which wins on raw OMR accuracy

Training data preparation:

For SFT:

1. Clean labeled OMR pairs (system-level):
    * GrandStaff (antoniorv6/grandstaff, ~14k piano system images + bekern labels): https://huggingface.co/datasets/antoniorv6/grandstaff — already used by SMT; direct reuse
    * PrIMuS / Camera-PrIMuS (~87k incipits; Camera variant adds blur/perspective artifacts): https://grfia.dlsi.ua.es/primus/
    * OLiMPiC (pianoform systems with MusicXML labels): https://github.com/ufal/olimpic-icdar24

2. Synthetic generation (unlimited scale):
    * Music source corpora (MusicXML or kern → render for free GT):
        - OpenScore (public domain MusicXML on HuggingFace): https://huggingface.co/OpenScore — Lieder corpus, string quartets, piano works
        - KernScores (humdrum kern; CCARH): https://kern.humdrum.org — large piano repertoire
        - MuseScore Hub (large but requires scraping; use with care re: licensing)
    * Rendering pipeline (now implemented): MusicXML/kern → `cvlization/lilypond:latest` → PNG
        - Produces pixel-perfect (image, kern) pairs with no annotation cost
        - System-level crops preferred (closer to SMT training distribution, smaller VLM context)
        - See `datasets/omr/deepscores_v2/` for pattern; equivalent `datasets/omr/openscore/` to build
    * Apply vintage augmentation on top — two approaches:
        A. Handcrafted transforms (fast baseline): paper texture, yellowing, foxing spots, ink fading, bleed-through, skew, scan noise via albumentations or custom PIL pipeline
        B. Style transfer augmentation (more authentic): use real vintage IMSLP scans as style references → transfer their look onto clean LilyPond renders. Same models as Bucket 4 Sub-task 2 (see below). More realistic than handcrafted transforms; converges with the user-facing rendering pipeline.
    * Approach B and Bucket 4 Sub-task 2 are the SAME technical problem — solving one gives the other for free

3. Instruction-tuning format (for VLM SFT):
    * Each example: system-prompt + image + "Transcribe this score to ekern notation" → ekern string
    * Use chat template of target model (Qwen3-VL uses standard messages format with image tokens)

For RL:

* No labeled output required — reward comes from structural rules applied to model output
* Reward signals (verifiable, no human annotation needed):
    1. Syntactic validity: is output valid ekern / MusicXML? (parse with music21 or xmllint)
    2. Duration correctness: do notes in each measure sum to the correct time signature?
    3. Voice consistency: no overlapping notes within the same voice
    4. Invariance reward: run augmented versions of the same image → outputs should agree (self-consistency across degradations)
* Training images for RL: unlabeled vintage scans from IMSLP — no labels needed, reward is computed purely from output structure
* Algorithm: GRPO (simpler than PPO, works well for structured output tasks; used by DeepSeek-R1)

Evaluation bottleneck — ground truth on vintage scans:

For clean labeled datasets (GrandStaff, PrIMuS) CER/SER is straightforward. For real vintage scans there is no ground truth, which constrains how experiments can be evaluated:

* Coarse metadata (key signature, time signature, era, publisher) can be verified against library catalog records — low cost, but only tests a fraction of OMR output
* Full transcription accuracy on vintage requires human expert correction — hours per page, hard to scale
* Proxy metrics (syntactic validity, duration correctness, self-consistency across augmentations) work without labels but measure structural plausibility, not exact correctness
* Practical strategy: use labeled datasets to measure transcription accuracy on clean input; use proxy metrics and small hand-corrected samples (~200 pages, via active learning) to evaluate vintage robustness

This bottleneck directly motivates the RL approach above — it sidesteps the label requirement entirely.

Strategies for building ground truth without heavy human expert effort (roughly ordered by cost):

Zero human effort:
* Cross-model consensus as pseudo-GT: run multiple architecturally diverse models (rule-based Audiveris + fine-tuned SMT + zero-shot VLMs); where they all agree on a token → high confidence pseudo-label; where they disagree → flag for review. Architectural diversity means correlated errors are unlikely.
* IMSLP clean edition → OMR → GT: many 1880s public-domain pieces were later re-engraved as clean modern editions on IMSLP. If the same piece exists as a clean typeset PDF, run SMT on it (CER ~4%) for effectively free GT, then compare vintage scan output against the clean edition transcription.
* Render-and-compare via VLM judge: render ekern output back to score image using Verovio, then ask a VLM "do these two images of measure N show the same notes?" Less brittle than pixel comparison because it asks for musical equivalence, not pixel equality.
* Error injection / degradation simulation: take clean scores + known GT, apply vintage-style degradation (ink bleed, yellowing, paper texture, skew). Gives calibrated evaluation of degradation sensitivity without new GT.

Light human effort (minutes per page):
* Symbolic feature GT: a non-expert verifies key signature, time signature, measure count, repeat signs, dynamics in ~5 minutes per page. Lightweight "partial GT" fast to collect, meaningful to evaluate against.
* Crowdsourcing simple tasks: tasks like "how many flats in the key signature?" or "is this note above or below the staff?" are answerable by anyone with basic music literacy. Aggregate via majority vote.
* Measure-level alignment: use omr-layout-analysis to segment both a clean modern edition and the vintage scan into systems; match corresponding measures. Clean edition GT applies at measure granularity.

Moderate human effort (targeted, not per-page):
* Active learning / disagreement sampling: run all models, compute pairwise disagreement score per measure, present only the top-K most contested measures to an expert for adjudication. Focuses expensive expert time on genuinely ambiguous cases.
* MIDI alignment: if a MIDI performance of the piece exists (Petrucci, Mutopia, score-following recordings), audio-to-score alignment gives pitch + duration GT. Doesn't cover beaming/stems/slurs but covers melodic accuracy.

Empirical findings from our comparison on vintage_score_1884.jpg:
* All four models (SMT, Gemini 3.1 Pro, Claude Opus, GPT-5.2 Pro) disagree on key signature — C major, G major, B♭ major all proposed. Cross-model consensus doesn't help here; coarse metadata from library catalog records would resolve it cheaply.
* Gemini 3.1 Pro produced the most musically usable transcription (clean syntax, correct waltz bass-chord-chord texture). Claude Opus identified structural sections (INTRO/WALTZ) and ornaments but stopped early. SMT transcribed the most structural detail (repeat signs, beaming) but with invalid syntax in the opening. GPT-5.2 Pro declined entirely.

Bootstrapping ground truth with frontier APIs:

Frontier VLMs (Gemini 2.5 Flash/Pro, GPT-4o, Claude Opus) can partially mitigate the bottleneck, but only for coarse metadata — not note-level transcription:

* What works: key signature, time signature, tempo/expression markings, instrumentation, era/publisher — hallucination is low for these coarse visual questions. Our qwen3-omr experiment confirmed this (era and dynamics correct; key signature wrong).
* What does not work: note-level transcription — frontier VLMs hallucinate articulations, ties, slurs, and accidentals that don't exist. MSU-Bench and MusiXQA benchmarks show purpose-built OMR models (SMT) significantly outperform them on CER. Not suitable for bootstrapping note-level ground truth.
* ABC notation reduces hallucination: NOTA (arXiv 2502.14893, Feb 2026) shows that representing notation as ABC text (rather than ekern/MusicXML) improves grounding. Worth considering for the ekern transcription prompt design.
* Cost is negligible: Gemini 2.5 Flash at ~$11–50 for 10k pages.

Practical two-tier bootstrapping strategy:
1. Frontier API (Gemini 2.5 Flash) → extract coarse metadata for 10k–100k vintage pages: key sig, time sig, tempo, era. Cheap, scalable, useful as weak supervision and evaluation metadata.
2. SMT → note-level pseudo-labels on the same pages, filtered by confidence.
3. Human correction (active learning) → fix the uncertain remainder (~200 pages).

Bucket 4 — facsimile:

Note: document restoration (deskew, denoise, etc.) is NOT a sub-task here. For OMR robustness on degraded vintage scans, the right approach is e2e training with vintage augmentation — not a separate preprocessing step. Restoration also conflicts with facsimile zone coordinates, which must reference original scan pixels.

Two sub-tasks:

1. Zone/layout detection + interactive viewer: locate systems, measures, and symbols in the image → emit MEI `<facsimile>` zone coordinates (ulx/uly/lrx/lry) that link semantic events back to pixel regions in the original scan. The output is an HTML viewer where the original scan is the visual base layer and semantic elements (notes, measures, voices) are overlaid as clickable SVG regions. Hovering a symbol highlights it and shows its semantic content; larger groups (measure, system) are selectable.
    * Semantic format: MEI (not MusicXML or ekern) — MEI natively supports @facs zone linkage; MusicXML has no equivalent
    * Zone detector: any object detection backbone (DINO, RT-DETR) fine-tuned on annotated score layouts; or Molmo 2 (best open-weight grounding/bbox output, see Bucket 3)
    * Implemented: omr-layout-analysis YOLOv8 (OLA v2.0) — detects systems, grand_staff, staves, system_measures, stave_measures; 5 grand staff systems correctly detected on vintage_score_1884.jpg in ~4s; pretrained weights ~40MB; CVlization example at examples/perception/optical_music_recognition/omr_layout_analysis/
    * Viewer rendering: Verovio (MEI → SVG overlay): https://github.com/rism-digital/verovio
    * Reference viewer: mei-friend: https://github.com/mei-friend/mei-friend
    * Proposed pipeline: omr-layout-analysis (grand_staff crops) → SMT system-level model (PRAIG/smt-grandstaff) → bekern per system → concatenate. Likely outperforms full-page model on vintage scans because each system crop is smaller and closer to the system-level training distribution.

2. Legibility-enhanced alternative rendering: vintage scores are often hard to read (faded ink, degraded paper, ornate typography). This sub-task generates an alternative version that preserves the vintage color palette and background texture but replaces degraded or ornate symbols with cleaner, more legible notation — a middle ground between the raw scan and a fully modernized engraving. Useful as a reading aid alongside the original.
    * Input: original scan pixels (style reference) + clean Verovio render of extracted semantics (content/structure)
    * Output: image with vintage aesthetic (paper color, texture, background) but clearer symbols
    * Note: this is a reference-image style transfer task — content from Verovio render, style from vintage scan

    Open-weight models (as of early 2026), in order of suitability:
    * QwenStyle (Jan 2026; built on Qwen-Image-Edit; content-preserving style transfer with reference images; state-of-the-art): https://github.com/witcherofresearch/Qwen-Image-Style-Transfer
    * TeleStyle (Jan 2026; same as QwenStyle + video support; lightweight): https://huggingface.co/Tele-AI/TeleStyle
    * FLUX.1 Kontext (Jun 2025; in-context multimodal editing with image+text prompts): https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
    * IP-Adapter (Style Only mode) + ControlNet (lightweight; structure from ControlNet, style from IP-Adapter reference image): https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter
    * InstantStyle-Plus (explicit 3-part decomposition: style injection + spatial preservation + semantic preservation): https://github.com/instantX-research/InstantStyle-Plus
    * CycleGAN (unpaired domain transfer; no reference image needed; simpler but less controllable)
    * NanoBanana = Gemini 2.5 Flash Image (Google API, NOT open weights — proprietary)
    * Reference: style_transfer_idea.md in this directory

    Convergence with training data: these same models serve as the vintage augmentation pipeline for Bucket 3 SFT training — solving one solves both.


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


What's been implemented (CVlization examples)

All examples are under examples/perception/optical_music_recognition/:

| Example | Bucket | Status | Notes |
|---|---|---|---|
| audiveris/ | B1 (rule-based) | Done | MusicXML output; fails badly on vintage_score_1884.jpg — spurious key changes, sparse notes |
| smt_omr/ | B2 (OMR transformer) | Done | ekern output; full-page model (PRAIG/smt-fp-grandstaff); CER 4.05% on clean GrandStaff sample; noisy on vintage scan |
| qwen3_vl/ | B2a (VLM zero-shot, local) | Done | Qwen3-VL-8B 4-bit; 5 structured prompts (key sig, time sig, era, dynamics, ekern) |
| vlm_omr/ | B2a (VLM zero-shot, API) | Done | LiteLLM wrapper; Gemini 2.5 Flash, Gemini 3.1 Pro, GPT-4o, GPT-5.2, GPT-5.2 Pro, Claude Sonnet 4.6, Claude Opus 4.6; omr_comparison/ subfolder with HTML report |
| omr_layout_analysis/ | B4 E1 (zone detection) | Done | omr-layout-analysis YOLOv8 (OLA v2.0); 5 classes; 5 grand staff systems detected on vintage waltz in ~4s |

Key findings from the multi-model comparison on vintage_score_1884.jpg (Biddle's Piano Waltz, 1884):
* Gemini 3.1 Pro: best ekern output — clean syntax, correct waltz bass-chord-chord texture, 36 measures; key wrong (G major)
* Claude Opus 4.6: best musical awareness — identifies INTRO/WALTZ sections, ornaments, key change; partial transcription (16 measures), key wrong (C major)
* SMT (full-page): most structural detail (repeat signs, beaming); invalid mixed-duration chords in opening; noisy
* Audiveris: essentially fails — sparse notes, multiple spurious key changes within one page, beat counts wrong
* GPT-5.2 Pro: declines to transcribe, citing resolution; useful signal that the scan is genuinely challenging
* All models disagree on key signature — C, G, B♭ major all proposed; music expert suggested C major → F major

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
* B2a: general VLM zero-shot (Qwen3-VL prompted, no training)
* B2b: general VLM SFT small (Donut fine-tuned on GrandStaff)
* B2c: general VLM SFT large (Qwen3-VL-8B fine-tuned on GrandStaff + synthetic vintage)
* B2d: general VLM RL (Qwen3-VL-8B + GRPO with musical correctness reward on unlabeled vintage)
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

Experiment group E — Facsimile pipeline

* E1: zone detection — measure IoU of predicted system/measure bounding boxes vs. manual annotations; validate MEI facsimile output by checking @facs linkage correctness in the interactive viewer
* E2: legibility-enhanced rendering — FLUX/ControlNet+LoRA conditioned on clean symbol structure; evaluate (a) vintage aesthetic preservation: FID and LPIPS vs. held-out vintage pages from same era; (b) legibility: OMR accuracy on generated image vs. original scan (improvement = better legibility); (c) human preference: does it feel vintage but readable?

Metrics:

* E1: zone IoU, MEI facsimile validity, viewer interaction correctness
* E2: FID, LPIPS, OMR accuracy delta (generated vs. original scan), human preference score

Expected outcome:

* E1 enables the “click a note, highlight the scan region” demo; E2 is the reading-aid differentiator — vintage look preserved, symbols legible.


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

