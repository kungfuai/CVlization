# Moshi (Kyutai) — full-duplex spoken dialogue

Real-time speech-to-speech dialogue with **[Kyutai Moshi](https://github.com/kyutai-labs/moshi)**.
A single streaming transformer collapses STT + LLM + TTS into one model;
the live server path achieves ~200 ms practical end-to-end latency.

| | |
|---|---|
| Best for | Real-time spoken dialogue (voice in → voice out) |
| Default model | `kyutai/moshiko-pytorch-bf16` (male voice "Moshiko"); `…moshika-pytorch-bf16` is the female "Moshika" variant |
| Language | English only (multilingual checkpoint targeted post 2026Q1) |
| Two modes | **predict** = batch (one wav in → one wav out); **serve** = WebSocket server for live dialogue |
| Image size | ~12 GB (PyTorch CUDA + moshi + audio libs) |
| Output | wav file (Moshi's spoken response) for predict; WebSocket audio frames for serve |

## Two ways to run

### Batch dialogue (one-shot)

Best for show-and-tell: feed Moshi an input wav (the "user turn") and get
back a wav (Moshi's response).

```bash
cvl run moshi build
cvl run moshi predict          # uses the bundled CVL sample, writes moshi_response.wav
# or, your own user-turn audio:
cvl run moshi predict -- --audio my_question.wav --output reply.wav
```

### Live dialogue (full-duplex server)

```bash
cvl run moshi serve            # listens on 0.0.0.0:8998
```

Connect from another shell or browser:

```bash
# upstream terminal client (mic + speaker):
pip install moshi && python -m moshi.client --host <server-host> --port 8998
# or use the upstream JS client in the kyutai-labs/moshi repo (clients/web/),
# which needs a TLS endpoint (the upstream docker-compose ships a Cloudflare
# tunnel sidecar for that).
```

## What to expect

- **First run**: downloads ~6 GB for the default Moshiko bf16 checkpoint
  (Mimi codec is bundled inside the same repo). Cached in
  `~/.cache/cvlization/huggingface` thereafter.
- **What predict does**: runs Kyutai's `moshi-inference` under the hood — one
  user-turn wav in, one Moshi-response wav out. The bundled sample is
  `zzsi/cvl::livetalk/example.wav` ("It's an amazing gift…"); Moshi's reply
  is dialogue-shaped (it tries to converse, not transcribe).
- **What serve does**: starts `python -m moshi.server` on port 8998.
  Designed for the upstream JS / terminal clients, not for cvl-run-from-cwd
  scripting; treat it as the "live demo" mode.
- **Output**: predict writes a wav file (defaults to `moshi_response.wav` in
  your cwd via `cvl run`). serve streams audio over WebSocket.
- **Runtime**: batch wall ~10–15 s on RTX PRO 6000 (cold container + cached
  weights). Default `--pad-input-to 15` gives Moshi a 15-second window to
  respond; set lower for faster turns, 0 to disable padding (response length
  then matches input length, which can truncate the reply mid-sentence).
- **Reply text**: predict.py prints Moshi's text-side output at the end
  ("=== Moshi reply (text): …") so you can read the transcript without
  scrolling past inference logs.
- **Live full-duplex**: real-time bidirectional via the WebSocket server
  (see "Live dialogue" below). Not exercised by the batch `predict` preset.

## Sample

**Input** — user-turn audio (~6 s, 16 kHz mono, auto-downloaded):

[`livetalk/example.wav`](https://huggingface.co/datasets/zzsi/cvl/resolve/main/livetalk/example.wav)

**Output** — `moshi_response.wav` (Moshi's spoken reply). Audio doesn't
embed in GitHub markdown, but the wav will play in any audio player.

## Model variants

```bash
# Moshika (female voice)
MOSHI_HF_REPO=kyutai/moshika-pytorch-bf16 cvl run moshi predict
MOSHI_HF_REPO=kyutai/moshika-pytorch-bf16 cvl run moshi serve

# Int8-quantized (experimental, smaller VRAM)
MOSHI_HF_REPO=kyutai/moshiko-pytorch-q8   cvl run moshi predict
```

Apple MLX (`kyutai/moshiko-mlx-*`) and Rust/Candle (`kyutai/moshiko-candle-*`)
variants exist upstream — this preset is the PyTorch path; use the MLX/Rust
implementations directly from the [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
repo if you need Apple-silicon or production-grade Rust latency.

## Notes

- Moshi's WebSocket server (port 8998) is not OpenAI-API-compatible — it
  speaks Moshi's own binary audio protocol. The upstream JS / Python clients
  in `kyutai-labs/moshi` know that protocol.
- License: model weights are CC-BY 4.0; this wrapper is the repo's standard
  license.
- Real-time mic-in / speaker-out requires actual audio devices, so the live
  demo can't be one-shotted by `cvl run moshi serve` alone — you also need
  a client. The batch `predict` path is the show-and-tell-friendly mode.
