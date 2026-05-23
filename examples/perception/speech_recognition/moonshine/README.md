# Moonshine ASR

Low-latency English speech recognition with **Useful Sensors Moonshine** —
tiny (~27M) and base (~61M) models optimised for on-device / real-time
transcription. Beats similarly-sized Whisper variants on WER and is roughly
5× faster on 10-second clips at the same accuracy.

## Why Moonshine vs Whisper / Parakeet

- **Moonshine** — on-device / low-latency, CPU-friendly, tiny / base only
- **faster-whisper** — Whisper-family quality, CTranslate2 backend, broad
  multilingual + translation support
- **parakeet-tdt** — server-side throughput champion (~2000x RTFx on GPU)

This preset is for the "fast and small, CPU is fine, English only" niche.

## Quick start

```bash
bash examples/perception/speech_recognition/moonshine/build.sh
bash examples/perception/speech_recognition/moonshine/predict.sh
# default: moonshine/base on CPU, sample audio, JSON output
```

Overrides:

```bash
# Tiny model (smaller / faster, slightly lower accuracy)
./predict.sh --model moonshine/tiny --audio /path/to.wav

# Use a GPU if available
USE_GPU=1 ./predict.sh --audio /path/to.wav

# Plain-text transcript instead of JSON
./predict.sh --audio sample --format txt --output transcript.txt
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
