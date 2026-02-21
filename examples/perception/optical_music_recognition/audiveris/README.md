# Audiveris — Classical OMR Baseline

Rule-based Optical Music Recognition using [Audiveris 5.9.0](https://github.com/Audiveris/audiveris). Takes a scanned sheet music image and outputs **MusicXML** (`.mxl`) — directly playable and editable in MuseScore, Sibelius, Finale, etc.

No GPU required. Useful as a **baseline** against learned OMR models such as `smt-omr`.

## Quick Start

```bash
./build.sh
./predict.sh                        # auto-downloads sample image (allegretto.png)
./predict.sh --image my_score.jpg   # your own score
```

> **Image requirements**: Audiveris needs ~300 DPI full-page scans. System-level crops or low-resolution images will be rejected with "resolution too low" warnings.

## Output format

Audiveris outputs **MusicXML 4.0** (`.mxl` compressed format). This can be:

- Opened directly in MuseScore, Sibelius, Finale, etc.
- Converted to MIDI for playback
- Processed with music21 or verovio

## How it works

Audiveris is a traditional multi-stage OMR pipeline:

```
LOAD → BINARY → SCALE → GRID → HEADERS → BEAMS → HEADS →
STEMS → REDUCTION → TEXTS → MEASURES → CHORDS → CURVES →
SYMBOLS → LINKS → RHYTHMS → PAGE → export MusicXML
```

Each step is a deterministic image processing or classification stage — no neural network training required. The TEXTS step uses **Tesseract OCR** for lyrics and tempo markings.

## Headless operation

Audiveris initialises Java AWT even in `-batch` mode. The `predict.py` wraps the call with `xvfb-run` to provide a virtual display inside Docker.

## Comparison with smt-omr

| | `audiveris` | `smt-omr` |
|-|-------------|-----------|
| Approach | Rule-based pipeline | End-to-end transformer |
| Output | MusicXML (.mxl) | ekern notation |
| GPU needed | No | Yes (~4 GB) |
| Vintage robustness | Moderate | Better (trained on real scans) |
| Correction workflow | Built-in GUI editor | Manual post-processing |

## References

- Repository: https://github.com/Audiveris/audiveris
- Handbook: https://audiveris.github.io/audiveris/
- Releases: https://github.com/Audiveris/audiveris/releases
