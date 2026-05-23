# Parakeet-TDT (NeMo) ASR

High-throughput English speech recognition with **NVIDIA NeMo Parakeet-TDT**
(FastConformer encoder + Token-and-Duration Transducer decoder). The TDT
decoder's blank-skipping path is what delivers the ~2000x RTFx headline on
the Hugging Face Open-ASR leaderboard.

## Why NeMo (not HF transformers)?

There's a transformers adapter for Parakeet, but it's a generic-transducer
port — most of TDT's speedup lives in NeMo's native decoder. Quality is the
same (same weights), but speed isn't. If the multi-GB NeMo install is a
non-starter, see "Transformers fallback" below.

## Quick start

```bash
cvl run parakeet-tdt build
cvl run parakeet-tdt predict      # transcribes the bundled CVL sample
```

Or the direct shell path:

```bash
bash examples/perception/speech_recognition/parakeet_tdt/build.sh
bash examples/perception/speech_recognition/parakeet_tdt/predict.sh \
  --audio sample \
  --output transcript.json
```

Defaults: `nvidia/parakeet-tdt-1.1b`, auto-device (CUDA if visible), JSON output.

## What to expect

- **First run**: downloads ~4.5 GB of model weights from Hugging Face
  (`nvidia/parakeet-tdt-1.1b`) into the shared HF cache (~/.cache/cvlization).
  Cached on subsequent runs.
- **What it does**: transcribes one audio file. Defaults to the bundled
  ~6-second 16 kHz mono CVL sample (`zzsi/cvl::livetalk/example.wav`); pass
  `--audio /path/to.wav` for your own. Non-16-kHz / non-mono inputs are
  auto-resampled (librosa) to a sibling `*.16k.wav`.
- **Output**: a JSON file (default `parakeet_tdt_transcript.json`) in your
  current directory when run via `cvl run`. Fields: `text` (lowercase
  transcript, as Parakeet emits no casing), `model`, `device`, `audio`,
  `task`, `created_at`, and `segments` if `--word-timestamps` is passed.
  `--format txt` and `--format srt` (with timestamps) also supported.
- **Runtime** on RTX PRO 6000 (single GPU): ~10 s warm / ~50 s with NeMo's
  startup import on a cached model. The transcription itself is well under
  a second for short clips.

## Sample

**Input** — bundled CVL audio clip (~6 s, 16 kHz mono, auto-downloaded):

[`livetalk/example.wav`](https://huggingface.co/datasets/zzsi/cvl/resolve/main/livetalk/example.wav)

**Output** — JSON transcript:

```json
{
  "text": "it's an amazing gift and a unique privilege and i love going",
  "model": "nvidia/parakeet-tdt-1.1b",
  "device": "cuda",
  "task": "transcribe"
}
```

## Model overrides

```bash
# Lighter / newer 0.6B variant (often faster + comparable accuracy)
./predict.sh --model nvidia/parakeet-tdt-0.6b-v2 --audio /path/to.wav

# Multilingual TDT (0.6B-v3)
./predict.sh --model nvidia/parakeet-tdt-0.6b-v3 --audio /path/to.wav

# Word timestamps in JSON
./predict.sh --audio sample --word-timestamps --format json
```

Output formats: `json` (default; includes segments if `--word-timestamps`),
`txt` (transcript text only), `srt` (needs `--word-timestamps`).

## Audio requirements

Parakeet expects **16 kHz mono**. Inputs at other sample rates / multi-channel
are auto-resampled (via librosa) to a sibling `*.16k.wav` before inference.

## Transformers fallback (lean, slower)

```bash
pip install transformers torchaudio
# In a Python shell:
from transformers import AutoModelForCTC, AutoProcessor   # adapter API
```

Use the transformers path only when the NeMo install is impractical (e.g.
edge devices). Quality matches; throughput typically lags the NeMo path
because TDT's blank-skipping isn't fully ported.

## Notes

- NeMo's startup is noisy by default (hundreds of import/config log lines).
  This preset silences stdout during import and model load. Pass `--verbose`
  to restore the noise.
- HF cache is mounted from the host (`~/.cache/cvlization/huggingface`) and
  shared with other CVL examples — Parakeet checkpoints download once.
- Default model is **`nvidia/parakeet-tdt-1.1b`** (the 1.1B variant noted as
  "fastest server-side option" with >2000 RTFx). The 0.6B-v2 is often a
  better speed/accuracy trade-off in practice; try both.
