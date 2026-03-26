# Sheet Music Transformer (SMT) — Optical Music Recognition

End-to-end Optical Music Recognition (OMR) for pianoform sheet music. Takes a scanned score image and outputs **ekern notation** (`**ekern_1.0`) — a structured encoding of the musical content (pitches, durations, voices, dynamics, and structure) compatible with the Humdrum **kern format.

## Sample

**Input** — system-level piano scan (auto-downloaded from HuggingFace):

![Sample score](https://huggingface.co/datasets/zzsi/cvl/resolve/main/smt_omr/sample_score.jpg)

**Output** — ekern notation (CER 0.28% vs ground truth):

```
**ekern_1.0	**ekern_1.0
*clefF4	*clefG2
*k[b-]	*k[b-]
*M2/4	*M2/4
=-	=-
8FL	16r
.	16ffLL
8cJ 8A	16ee
.	16ddJJ
8CL	16ccLL
.	16bn
8cJ 8A	16cc
.	16aJJ
=	=
8GL	8.aaL 8.b-
8eJ 8c 8B-	.
...
*-	*-
```

Each line is a beat position. The two tab-separated columns are bass staff | treble staff, compatible with Humdrum **kern.

## Models

Use models from the [PRAIG collection](https://huggingface.co/collections/PRAIG/sheet-music-transformer-6853c4ca1bd7980a91677dfd) — these match the current SMT codebase weight naming.

| Model | Scope | Notes |
|-------|-------|-------|
| `PRAIG/smt-grandstaff` | System-level piano | Default; clean and camera scans |
| `PRAIG/smt-fp-grandstaff` | Full-page piano | Multi-system pages |
| `PRAIG/smt-fp-mozarteum` | Full-page (Mozarteum) | Specific corpus |
| `PRAIG/smt-fp-polish-scores` | Full-page (Polish) | Specific corpus |

> **Note**: The older `antoniorv6/` checkpoints use a different weight naming convention and are incompatible with the current SMT codebase.

## Quick Start

```bash
# Build the Docker image
./build.sh

# Run inference (downloads a sample image automatically)
./predict.sh

# Run inference on your own score
./predict.sh --image my_score.jpg

# Fine-tune (smoke test: 10 steps on 50 samples)
./train.sh
```

## Inference

```bash
# Auto-download a sample from HuggingFace and transcribe it
./predict.sh

# Use the full-page model variant
./predict.sh --model PRAIG/smt-fp-grandstaff --image score.jpg

# Save as JSON with metadata
./predict.sh --image score.png --format json --output result.json
```

### Output format

SMT is trained on **bekern** — a token encoding where `·` inter-token separators and whitespace are replaced with explicit tokens (`<b>` = newline, `<s>` = space/dot, `<t>` = tab). `predict.py` decodes these tokens back to readable whitespace, producing **ekern** (`**ekern_1.0`) notation in the output file. The `·` dots do not appear in the output — they become spaces.

The output is a two-column tab-separated file (bass staff | treble staff) that is compatible with the [**kern](https://www.humdrum.org/guide/ch02/) format and can be further processed with Verovio (SVG rendering) or music21 (MusicXML export).

### Accuracy metrics

When running with the auto-downloaded sample (no `--image` flag), accuracy metrics are computed automatically against the bundled ground truth:

```
CER (Character Error Rate): character-level edit distance
SER (Symbol Error Rate):    space-separated token edit distance
LER (Line Error Rate):      line-level edit distance
```

To evaluate against your own ground truth:
```bash
./predict.sh --image score.jpg --ground-truth score_gt.txt
```

## Fine-tuning

Edit `config.yaml` to configure the training run, then:

```bash
./train.sh
```

For full training (not smoke test):
```yaml
# config.yaml
dataset:
  max_samples_train: null  # use all ~10k training samples
  max_samples_val: null
training:
  max_steps: 10000
  wandb_offline: false     # enable online W&B logging
```

The fine-tuner loads the pretrained SMT vocabulary (baked into the model config), so it is compatible with the GrandStaff dataset out of the box.

## VRAM requirements

| Task | Approximate VRAM |
|------|-----------------|
| Inference (fp16) | ~2–4 GB |
| Fine-tuning (batch 1, fp16) | ~6–8 GB |

## References

- Paper (system-level): [Sheet Music Transformer (ICDAR 2024)](https://arxiv.org/abs/2402.07596)
- Paper (full-page): [End-to-End Full-Page OMR](https://arxiv.org/abs/2405.12105)
- Repository: https://github.com/antoniorv6/SMT
- Models: https://huggingface.co/collections/PRAIG/sheet-music-transformer-6853c4ca1bd7980a91677dfd
- Dataset: https://huggingface.co/datasets/antoniorv6/grandstaff
