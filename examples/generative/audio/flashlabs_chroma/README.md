# FlashLabs Chroma-4B — voice-cloning spoken dialogue

Turn-based spoken dialogue with **optional voice cloning** via
[FlashLabs Chroma-4B](https://huggingface.co/FlashLabs/Chroma-4B). Architecture:
Qwen2.5-Omni-3B reasoner + Llama3 backbone (16 layers, 2048 hidden) + Llama3
decoder (4 layers, 1024 hidden) + Mimi codec (24 kHz). Apache-2.0 weights.

| | |
|---|---|
| Best for | Voice cloning + consumer-GPU spoken dialogue |
| Default model | `FlashLabs/Chroma-4B` (~8 GB weights) — **HF-gated, see below** |
| Min VRAM | ~8 GB at bf16, **~4 GB at 4-bit** (`CHROMA_QUANT=4bit`, bitsandbytes) |
| Language | English only |
| Mode | Turn-based (record → process → reply); NOT continuous full-duplex |
| Output | 24 kHz mono PCM16 wav |

## How it differs from Moshi (sibling preset)

| | Moshi | Chroma |
|---|---|---|
| Streaming | ✅ Real-time full-duplex | ❌ Turn-based |
| Voice cloning | ❌ Fixed Moshiko/Moshika | ✅ Reference audio → cloned voice |
| Min VRAM | ~18 GB | ~4 GB (4-bit) |
| Inference API | Custom WebSocket server | Plain HF transformers |
| Live demo | Browser at `localhost:8998` (one command) | File-in / file-out (this preset). Upstream's `local_voice_chat.py` does mic/speaker via pyaudio. |

Pick Moshi for natural continuous conversation; pick Chroma when you want
to clone a voice or fit on a 4–8 GB consumer card.

## HF gating — read this before running

`FlashLabs/Chroma-4B` is auto-gated. First-run requirements:

1. Visit [huggingface.co/FlashLabs/Chroma-4B](https://huggingface.co/FlashLabs/Chroma-4B), accept the gate (one click; auto-approved).
2. Create an HF access token at huggingface.co/settings/tokens.
3. Export it: `export HF_TOKEN=hf_...` before running `cvl run flashlabs-chroma ...`.

`predict.sh` forwards `HF_TOKEN` into the container; once weights are
cached locally you don't need the token again.

## Quick start

```bash
cvl run flashlabs-chroma build
HF_TOKEN=hf_... cvl run flashlabs-chroma predict     # bundled CVL sample, self-clone
```

Or the direct shell path:

```bash
bash examples/generative/audio/flashlabs_chroma/build.sh
HF_TOKEN=hf_... bash examples/generative/audio/flashlabs_chroma/predict.sh \
  --audio sample --output reply.wav
```

## What to expect

- **First run**: ~8 GB model weights pulled to `~/.cache/cvlization/huggingface`.
  HF_TOKEN required only on first run (and only because of the gate).
- **What it does**: takes one user-turn audio file + a voice-clone reference
  audio file, produces one response audio file in the cloned voice. Default
  is self-clone (reference = the user's own audio), which avoids shipping
  any specific person's voice as the default.
- **Output**: 24 kHz mono PCM16 wav, default `chroma_response.wav` in your cwd.
- **Runtime** on RTX PRO 6000 (single GPU): ~20–30 s wall (container + model
  load + inference) for a short reply. Inference itself is several seconds
  per turn — not real-time-streaming.

## Voice cloning — overriding the reference

```bash
# Clone someone else's voice: provide a reference recording of them speaking
HF_TOKEN=hf_... cvl run flashlabs-chroma predict -- \
  --audio my_question.wav \
  --prompt-audio reference_voice.wav \
  --prompt-text "transcript of reference_voice.wav, helps the conditioner"

# Lower-VRAM 4-bit (bitsandbytes) — runs on ~4 GB consumer cards
HF_TOKEN=hf_... CHROMA_QUANT=4bit cvl run flashlabs-chroma predict
```

> **Ethical note**: voice cloning of real people without consent is a known
> deployment hazard. The upstream repo ships celebrity reference audio
> (Ariana Grande / Trump / LeBron / Scarlett Johansson) in `example/prompt_audio/`;
> this preset deliberately does **not** wire those in as defaults. Use
> recordings you have permission to use.

## Sample

**Input** — bundled CVL audio clip (~6 s, auto-downloaded):

[`livetalk/example.wav`](https://huggingface.co/datasets/zzsi/cvl/resolve/main/livetalk/example.wav)

Used as both the user query and the voice-clone reference (self-clone).

**Output** — `chroma_response.wav` (Chroma's spoken reply in the cloned
voice). Audio doesn't embed in GitHub markdown; play it in any audio player.

## Notes

- Upstream pins `transformers==5.0.0rc0` (release-candidate); this preset
  honours that pin. Newer transformers versions may work via
  `trust_remote_code=True` but haven't been validated.
- The `chroma` name collides with [`chromadb`](https://github.com/chroma-core/chroma)
  (vector DB). This preset is named `flashlabs-chroma` in cvl-info to
  disambiguate.
- Project is newer / smaller than Kyutai's Moshi (`stability: experimental`).
- For a live mic+speaker demo on your local machine, see upstream's
  `local_run/chroma_4bit_pack/local_voice_chat.py` — it uses pyaudio and
  reads from a host mic, which doesn't play nicely with our containerised
  setup. This preset is the file-in / file-out alternative.
