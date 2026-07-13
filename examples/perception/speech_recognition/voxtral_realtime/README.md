# Voxtral Mini 4B Realtime — Multilingual Streaming ASR

Realtime multilingual speech transcription powered by
[Mistral Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602),
served via [vLLM](https://github.com/vllm-project/vllm)'s `/v1/realtime` WebSocket endpoint.

## Why

CVlization already has file-based low-latency (Moonshine) and high-throughput (Parakeet)
ASR examples, but no streaming transcription service. Voxtral fills this gap with:

- **Realtime WebSocket streaming** — audio is transcribed as it arrives, not after upload
- **13 languages** — Arabic, German, English, Spanish, French, Hindi, Italian, Dutch, Portuguese, Chinese, Japanese, Korean, Russian
- **Configurable latency** — 80ms to 2400ms transcription delay (recommended: 480ms)
- **Production-grade serving** — vLLM backend with OpenAI-compatible realtime API
- **Apache 2.0 license** — free for research and commercial use

## What to Expect

| Item | Detail |
|------|--------|
| First-run model download | ~8 GB (cached in `~/.cache/huggingface`) |
| Docker image size | ~40 GB (PyTorch 2.11 + CUDA 13 + vLLM 0.25) |
| Server startup time | 60–180s (model loading) |
| Transcription latency | <500ms per utterance at default settings |
| Output location | `./voxtral_realtime_transcript.json` in your working directory |
| Output format | JSON with transcript text, incremental deltas, timing, and usage stats |

## Sample

**Input** — 5-second English speech (auto-downloaded from
[zzsi/cvl](https://huggingface.co/datasets/zzsi/cvl/blob/main/voxtral_realtime/example_en.wav)):

A 16 kHz mono WAV clip: _"It's an amazing gift and a unique privilege and I love going."_

**Output** — streaming transcript with incremental deltas
([sample_output.json](https://huggingface.co/datasets/zzsi/cvl/blob/main/voxtral_realtime/sample_output.json)):

```json
{
  "text": " It's amazing gift and a unique privilege. And I love going",
  "deltas": ["", "", " It", "'s", " amazing", " gift", " and", " a", " unique", " privilege", ".", " And", " I", " love", " going"],
  "audio_duration_sec": 5.0,
  "timing": {
    "total_sec": 7.94,
    "stream_sec": 0.0,
    "transcription_sec": 1.15
  }
}
```

**French input** — 5-second TTS clip
([example_fr.wav](https://huggingface.co/datasets/zzsi/cvl/blob/main/voxtral_realtime/example_fr.wav)):
_"Bonjour, comment allez-vous aujourd'hui? C'est une belle journée pour la science."_

**French output**
([sample_output_fr.json](https://huggingface.co/datasets/zzsi/cvl/blob/main/voxtral_realtime/sample_output_fr.json)):

```
Bonjour, comment allez-vous aujourd'hui? C'est une belle journée pour la science.
```

Tested on RTX PRO 6000 Blackwell (98 GB VRAM). Second run completes in ~1.6s.

## Quick Start

```bash
# 1. Build the Docker image
bash build.sh

# 2. Start the vLLM realtime server (runs detached, needs 16+ GB VRAM)
bash serve.sh

# 3. Wait for the server to load the model
until curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 5; done

# 4. Run the streaming transcription client
bash predict.sh

# 5. Stop the server when done
bash stop.sh
```

Or run the full automated cycle:

```bash
bash test.sh
```

### Using the CVL CLI

From any directory (requires `pip install cvlization`):

```bash
cvl run voxtral_realtime build
cvl run voxtral_realtime serve
# Wait for model to load (~100s)
until curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 5; done
cvl run voxtral_realtime predict
cvl run voxtral_realtime stop
```

`cvl run voxtral_realtime predict` mounts your current directory and saves
output to `./voxtral_realtime_transcript.json`.

## Options

### Server (`serve.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `mistralai/Voxtral-Mini-4B-Realtime-2602` | HuggingFace model ID |
| `PORT` | `8000` | Server port |
| `VLLM_MAX_MODEL_LEN` | `45000` | Max context (~1 hour audio) |
| `VLLM_EXTRA_ARGS` | _(empty)_ | Additional vLLM server flags |
| `VOXTRAL_DETACH` | `1` | Run detached (`0` = foreground) |
| `VOXTRAL_CONTAINER_NAME` | `cvl-voxtral-realtime-server` | Docker container name |

### Client (`predict.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--audio` | `sample` | Audio file path or `sample` for the default CVL sample |
| `--host` | `localhost` | vLLM server host |
| `--port` | `8000` | vLLM server port |
| `--model` | `mistralai/Voxtral-Mini-4B-Realtime-2602` | Model ID |
| `--chunk-size` | `4096` | Audio chunk size in bytes |
| `--output` | `voxtral_realtime_transcript.json` | Output file name |
| `--format` | `json` | Output format (`json` or `txt`) |
| `--verbose` | _(off)_ | Show detailed event logging |

## Latency / Quality Tradeoffs

Voxtral exposes a transcription delay parameter (configurable via `--max-model-len`):

| Delay | Use Case | Approx. WER (English) |
|-------|----------|----------------------|
| 80ms | Ultra-low latency, live captions | Higher |
| 480ms | **Recommended** balance | ~7% (GigaSpeech) |
| 2400ms | Maximum accuracy, batch-like | Lowest |

One text token corresponds to 80ms of audio. Increasing `max_model_len` allows
longer recordings (45000 tokens ~ 1 hour).

## Supported Languages

Arabic, Chinese, Dutch, English, French, German, Hindi, Italian, Japanese,
Korean, Portuguese, Russian, Spanish.

## Model Caching

Model weights are cached in `~/.cache/huggingface/` (mounted into the container).
The first run downloads ~8 GB; subsequent runs reuse the cache with no download.

## Architecture

```
┌─────────────┐         WebSocket          ┌──────────────────┐
│ predict.py  │  ────────────────────────►  │  vLLM Server     │
│  (client)   │  PCM16 audio chunks         │  /v1/realtime    │
│             │  ◄────────────────────────  │                  │
│             │  transcription.delta events  │  Voxtral 4B      │
└─────────────┘                             └──────────────────┘
```

The server loads the Voxtral model into GPU VRAM and exposes a WebSocket endpoint.
The client converts audio to PCM16 @ 16kHz, streams it in 4KB chunks, and receives
incremental transcription events as they are generated.

## Output Example

```json
{
  "text": "It's an amazing gift and a unique privilege and I love going.",
  "deltas": ["It's", " an", " amazing", " gift", ...],
  "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
  "audio_duration_sec": 3.5,
  "timing": {
    "total_sec": 2.1,
    "stream_sec": 0.3,
    "transcription_sec": 1.8
  }
}
```

## GPU Requirements

- **Minimum**: 16 GB VRAM (e.g., NVIDIA T4, RTX 4080)
- **Recommended**: 24+ GB VRAM (e.g., RTX 4090, A10, L4)
- Model weights: ~8 GB (4B params in BF16)
- vLLM pre-allocates remaining VRAM for KV cache (configurable via
  `VLLM_GPU_MEMORY_UTILIZATION`). On a 98 GB GPU with max_model_len=45000 the
  server may report ~92 GB allocated; on a 16 GB GPU it will fill ~14 GB,
  leaving headroom for the OS

## References

- [Model card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [Mistral announcement](https://mistral.ai/news/voxtral-transcribe-2)
- [vLLM documentation](https://docs.vllm.ai/)
- [Technical report](https://arxiv.org/abs/2602.11298)
