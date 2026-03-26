# Qwen3-VL — Zero-Shot OMR Probe

Zero-shot Optical Music Recognition probing using [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). Sends 5 music-theory prompts to the model on a single sheet-music image and records all responses.

This is the **B2a baseline** (zero-shot large VLM, no training) in the Bucket 3 model comparison matrix defined in [`plan.md`](../plan.md).

No fine-tuning. GPU required (~8 GB VRAM with 4-bit quantization).

## Sample

The default sample is **Biddle's Piano Waltz** (Robert D. Biddle, 1884), from the [Library of Congress American Sheet Music collection](https://www.loc.gov/item/2023852770/). Public domain.

**Cover page** (`--sample-cover`) — ornate Victorian typography, library stamps, visible aging:

![Vintage cover](https://huggingface.co/datasets/zzsi/cvl/resolve/main/qwen3_omr/vintage_cover_1884.jpg)

**Music notation page** (default) — grand staff, 3/4 waltz, Eb major, 19th-century engraving:

![Vintage score](https://huggingface.co/datasets/zzsi/cvl/resolve/main/qwen3_omr/vintage_score_1884.jpg)

**Output** — actual results from a verified run:

```
[key_signature] Key signature
Response: "one sharp (the F#) → F# major"
→ WRONG — the model misread flats as a sharp. Key signature reading is unreliable zero-shot.

[time_signature] Time signature
Response: "3/4 — three quarter-note beats per measure (standard waltz time)"
→ CORRECT ✓

[musical_era] Musical era
Response: "Romantic era — late 1880s. Copyright 1884, J. Biddle & Son. Typography and
engraving style consistent with 19th-century American parlor music."
→ EXCELLENT ✓ — the model read the copyright text and dated the piece precisely.

[dynamics_tempo] Dynamics and tempo markings
Response: "pp (intro, m.1), f (intro, last measure), cres. (before f), p (waltz, m.1),
ff (waltz, glissando section), legato p; Andante (intro), ben marcato, WALTZ, marcato,
glossando [sic]"
→ GOOD ✓ — most markings correctly identified; "glossando" misspelling of glissando.

[ekern_transcription] Full ekern transcription
Response: repeats "*clefG2*clefF4 / *M3/4 / *kF" ~25 times in a loop; no actual notes.
→ FAILED as predicted — hallucination loop, no tab-separated format, no note content.
```

**Key takeaway**: Coarse text reading (era, publisher, tempo markings) works well. Symbol counting (key signature) fails. Full transcription completely fails. This is the expected B2a lower-bound result.

## Quick Start

```bash
./build.sh
./predict.sh                            # music notation page (default)
./predict.sh --sample-cover             # decorative cover page
./predict.sh --image my_score.jpg       # your own vintage score
./predict.sh --prompts key_signature time_signature   # run subset only
```

## Prompts

| ID | Label | Expected quality |
|----|-------|-----------------|
| `key_signature` | Key signature | Good — visually salient |
| `time_signature` | Time signature | Good — visually salient |
| `musical_era` | Musical era | Moderate |
| `dynamics_tempo` | Dynamics/tempo markings | Moderate |
| `ekern_transcription` | Full ekern transcription | Poor — lower-bound probe |

## Options

```bash
# Use a smaller model (less VRAM)
./predict.sh --model Qwen/Qwen3-VL-2B-Instruct

# Disable 4-bit quantization (needs more VRAM, higher quality)
./predict.sh --no-quantize

# Run only selected prompts
./predict.sh --prompts key_signature time_signature musical_era

# Custom output path
./predict.sh --output /tmp/my_probe.txt
```

## VRAM requirements

| Setting | Approx. VRAM |
|---------|-------------|
| 8B with 4-bit quantization (default) | ~8 GB |
| 8B in bf16 (no quantization) | ~16 GB |
| 2B with 4-bit quantization | ~4 GB |

## Role in the experiment plan

This example establishes the **B2a zero-shot lower bound** for the Bucket 3 comparison:

| | Small model | Large model |
|--|-------------|-------------|
| Zero-shot | — | **Qwen3-VL prompted (this example)** |
| SFT | Donut fine-tuned | Qwen3-VL-8B fine-tuned on OMR data |
| RL | optional | GRPO with musical correctness reward |

The zero-shot probe answers the question: *"What can a large VLM do with no music-specific training at all?"*

## Comparison with smt-omr

| | `qwen3-omr` (zero-shot) | `smt-omr` |
|--|-------------------------|-----------|
| Training | None | Trained on GrandStaff (~14k) |
| Sample image | Vintage 1884 LoC scan | Clean GrandStaff system crop |
| Output | Q&A text | ekern notation |
| Key/time sig | Good | Implicit in output |
| Vintage robustness | Unknown (key test goal) | Unknown on degraded scans |
| Full transcription | Hallucinated | CER ~0.28% on clean input |
| GPU VRAM | ~8 GB | ~4 GB |

## References

- Model: [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- Paper: [Qwen3-VL Technical Report](https://arxiv.org/abs/2504.10479)
- SMT comparison: [smt_omr/](../smt_omr/)
