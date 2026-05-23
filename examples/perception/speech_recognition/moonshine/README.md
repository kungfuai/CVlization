# Moonshine ASR

Low-latency English speech recognition with **Useful Sensors Moonshine** —
tiny (~27M) and base (~61M) models optimised for on-device / real-time
transcription. Beats similarly-sized Whisper variants on WER and is roughly
5× faster on 10-second clips at the same accuracy.

| | |
|---|---|
| Best for | On-device / CPU / low-latency |
| Default model | `moonshine/base` (~130 MB); `moonshine/tiny` (~85 MB) also available |
| Language | English only |
| Latency, this preset | ~12 s wall on a single x86 CPU (cold container + Keras import + inference); inference itself is sub-second |
| Image size | ~9.9 GB (driven by `useful-moonshine`'s torch CUDA wheels — only the runtime libs are large) |
| Output | text transcript; JSON / TXT |

## Why Moonshine vs Whisper / Parakeet

- **Moonshine** — on-device / low-latency, CPU-friendly, tiny / base only
- **faster-whisper** — Whisper-family quality, CTranslate2 backend, broad
  multilingual + translation support
- **parakeet-tdt** — server-side throughput champion (~2000x RTFx on GPU)

This preset is for the "fast and small, CPU is fine, English only" niche.

## Quick start

```bash
cvl run moonshine build
cvl run moonshine predict          # transcribes the bundled CVL sample
```

Or the direct shell path:

```bash
bash examples/perception/speech_recognition/moonshine/build.sh
bash examples/perception/speech_recognition/moonshine/predict.sh
# default: moonshine/base on CPU, sample audio, JSON output
```

## What to expect

- **First run**: downloads ~130 MB for `moonshine/base` (or ~85 MB for
  `moonshine/tiny`) into the shared HF cache (~/.cache/cvlization). Cached
  thereafter.
- **What it does**: transcribes one audio file. Defaults to the bundled
  ~6-second 16 kHz mono CVL sample (`zzsi/cvl::livetalk/example.wav`); pass
  `--audio /path/to.wav` for your own. Non-16-kHz / non-mono inputs are
  auto-resampled (librosa).
- **Output**: a JSON file (default `moonshine_transcript.json`) in your
  current directory when run via `cvl run`. Fields: `text`, `model`,
  `audio`, `task`, `created_at`. `--format txt` also supported.
- **Runtime**: ~12 s wall end-to-end on a single x86 CPU on the bundled
  ~6 s clip (container spin-up + Keras/torch import + inference). Inference
  itself is well under a second — useful budget for real-time / on-device.

## Sample

**Input** — bundled CVL audio clip (~6 s, 16 kHz mono, auto-downloaded):

[`livetalk/example.wav`](https://huggingface.co/datasets/zzsi/cvl/resolve/main/livetalk/example.wav)

**Output** — JSON transcript (`moonshine/base`):

```json
{
  "text": "It's a amazing gift and a unique privilege and i love going",
  "model": "moonshine/base",
  "task": "transcribe"
}
```

> Note the literal `"It's a amazing"` — `moonshine/base` occasionally drops
> short function words (here, the article "an"). It's a known size/quality
> tradeoff of the smaller model family; `faster-whisper` and `parakeet-tdt`
> get this clip word-perfect. If you need higher accuracy and have a GPU,
> reach for `parakeet-tdt`; if you need lower latency on commodity CPU,
> Moonshine is still the right choice.

Overrides:

```bash
# Tiny model (smaller / faster, slightly lower accuracy)
cvl run moonshine predict -- --model moonshine/tiny --audio /path/to.wav

# Use a GPU if available
USE_GPU=1 cvl run moonshine predict -- --audio /path/to.wav

# Plain-text transcript instead of JSON
cvl run moonshine predict -- --audio sample --format txt --output transcript.txt
```

## Audio requirements

Moonshine expects **16 kHz mono**. Inputs at other sample rates / multi-channel
are auto-resampled (via librosa) to `*.16k.wav` before inference.

## Backend

`useful-moonshine` runs on Keras 3, which means the backend is selectable via
`KERAS_BACKEND` (`torch` | `jax` | `tensorflow`). This preset defaults to
**`torch`** so it matches the rest of the CVL stack and doesn't pull in a
separate JAX/TF install. Override by setting `KERAS_BACKEND=jax` (and the
matching extras) at build time.

## Notes

- Image is a `python:3.11-slim` base + `useful-moonshine` — small and fast to
  build (no CUDA toolkit in the image).
- HF cache is mounted from the host so model weights (downloaded by Moonshine's
  `transcribe()` on first run) sit alongside other CVL examples.
- Only English models are publicly available today (`moonshine/tiny`,
  `moonshine/base`). Multilingual variants haven't been released.
