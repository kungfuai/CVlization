# Per-measure VLM v4 — fresh from Qwen3-VL-8B base + active-header prompt

## Setup

- Data: `/tmp/per_measure_v4_merged/` from `build_per_measure_dataset.py`
  over the v3 detection sources (L7a 1000 train / 100 dev + L9 1000/100 +
  openscore 567/77 pages). Per-measure crops with 0.3 vertical pad so
  lyrics and dynamics below the staff land inside the box.
- 39,942 train / 4,387 dev per-measure samples.
- Base: `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` (4-bit), fresh
  LoRA r=16 alpha=16, finetune {vision, language, attention, mlp}.
- 1 epoch, batch 4 × grad_accum 4 = effective 16, lr 2e-5 cosine, max
  seq 1024. ~1h 40min wall clock on RTX Pro 6000.

## Prompt change vs prior runs

Old prompt was just "Transcribe this single measure of sheet music to
MXC2 ...". New prompt prepends the per-part active header parsed from
the GT MXC2:

```
Transcribe this single measure of sheet music to MXC2 ...

Active header per part:
P1 Voice: key=-4 time=4/4 clef=G2
P2 Piano RH: key=-4 time=4/4 clef=G2
P3 Piano LH: key=-4 time=4/4 clef=F4
```

Rationale: mid-system measures don't visually show key/time/clef, so the
model can't infer them from image alone. At inference the active header
should come from the detector's keysig prediction + part structure
known from the page-level layout.

## Loss curve

|step|train|eval|
|---|---|---|
|200|0.287|0.304|
|400|0.212|0.212|
|600|0.155|0.184|
|800|0.143|0.165|
|1000|0.140|0.159|
|1200|0.140|0.155|
|1400|0.130|0.151|
|1600|0.120|0.149|
|1800|0.115|0.147|
|2000|≈0.12|0.146|
|2200|≈0.12|0.145|
|2400|≈0.12|**0.145**|
|2497 (final)|—|**0.145**|

Plateau after ~step 1800. Eval-train gap ~0.02, no overfit.

## Checkpoint

`outputs/per_measure_fresh_v4/final_model/` (LoRA adapter on top of
`unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`).

## Open

- Not yet plugged into `pipeline_per_measure.py` for end-to-end eval
  against the v3 detector.
- Eval metric is just cross-entropy; need character-level / pitched
  similarity metric for honest task evaluation.
- The "active header from MXC2" trick uses GT at training; at inference
  the detector's keysig is the upstream signal. May need calibration if
  detector keysig differs from MusicXML truth.
