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
bash examples/perception/speech_recognition/parakeet_tdt/build.sh
bash examples/perception/speech_recognition/parakeet_tdt/predict.sh \
  --audio sample \
  --output outputs/transcript.json
```

Defaults: `nvidia/parakeet-tdt-1.1b`, auto-device (CUDA if visible), JSON output.

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
