# Qwen3-Omni-30B-A3B-Instruct — multimodal spoken dialogue

Turn-based multimodal dialogue (audio / image / video / text in,
**text + 24 kHz audio out**) via
[`Qwen/Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct).
30B-total / ~3B-active Mixture-of-Experts with three baked-in voices
(Ethan, Chelsie, Aiden). Apache-2.0; not gated.

| | |
|---|---|
| Best for | Multimodal (image/video/audio) chat with spoken replies |
| Default model | `Qwen/Qwen3-Omni-30B-A3B-Instruct` (bf16, ~60 GB on disk) |
| Min VRAM | ~79 GB bf16 at 15 s ctx; ~108 GB bf16 at 60 s ctx (single A100/H100-80 is borderline; 2× 80 GB or H200 for longer) |
| 4-bit option | `cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit` is wired in (`--quant 4bit`) but **currently produces garbled output** under transformers 5.9 + compressed-tensors 0.11/0.12 — see [AWQ-4bit known issue](#awq-4bit-known-issue) |
| Languages | 19 input speech langs; 10 output speech langs (English among them) |
| Mode | Turn-based (file in / text + file out); NOT real-time streaming |
| Output | 24 kHz mono PCM float wav + plain-text reply |

## How it compares to the sibling audio presets

| | Moshi | flashlabs-chroma | **qwen3-omni** |
|---|---|---|---|
| Modalities in | audio | audio | **audio + image + video + text** |
| Modalities out | audio + text | audio (text track internal) | **audio + text** |
| Voice cloning | ❌ fixed | ✅ reference audio | ❌ fixed (3 voices) |
| Streaming | ✅ full-duplex | ❌ turn-based | ❌ turn-based |
| Min VRAM | ~18 GB | ~4 GB (4-bit) | ~80 GB (bf16; 4-bit broken) |
| License | CC-BY | Apache 2.0 | Apache 2.0 |
| Gated | no | auto-gated | **no** |
| Inference API | custom WebSocket | plain HF | HF + `qwen-omni-utils` |

Pick Moshi for continuous full-duplex; pick Chroma for voice cloning on a
cheap card; pick Qwen3-Omni when you need image/video understanding **and**
a spoken reply.

## Quick start

```bash
cvl run qwen3-omni build
cvl run qwen3-omni predict        # bf16, bundled sample, voice=Ethan
```

Or the direct shell path:

```bash
bash examples/generative/audio/qwen3_omni/build.sh
bash examples/generative/audio/qwen3_omni/predict.sh \
  --audio sample --output reply.wav --speaker Chelsie
```

## What to expect

- **First run**: ~60 GB bf16 weights pulled to
  `~/.cache/cvlization/huggingface`. The model is not gated, so no HF
  token is needed.
- **What it does**: takes one user-turn audio file, produces (1) a printed
  text reply (the "Thinker" output) and (2) one audio file in the chosen
  voice (the "Talker" + Code2Wav output).
- **Output**: 24 kHz mono float PCM wav, default
  `qwen3_omni_response.wav` in your cwd. Text reply is printed to stdout
  between visible borders.
- **Runtime** on a single RTX PRO 6000 (bf16, "Hi who are you?" prompt,
  256 max-new-tokens): ~10 s model load on cached weights + ~5–15 s
  generate, producing a ~5–10 s spoken reply.

## Choosing a voice

```bash
QWEN3_OMNI_SPEAKER=Chelsie cvl run qwen3-omni predict
# or:
bash examples/generative/audio/qwen3_omni/predict.sh --speaker Aiden
```

Three baked-in voices ship with the model: `Ethan`, `Chelsie`, `Aiden`.

## VRAM scaling (bf16)

Per the upstream Qwen3-Omni README:

| Audio context | Min VRAM (bf16) |
|---|---|
| 15 s | 78.85 GB |
| 30 s | 88.52 GB |
| 60 s | 107.74 GB |
| 120 s | 144.81 GB |

Single A100/H100-80 GB is borderline at the shortest context; for longer
clips you need 2× 80 GB or one H200 141 GB. Disable the Talker stage
(below) to save ~10 GB.

## AWQ-4bit known issue

The community quant
[`cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit`](https://huggingface.co/cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit)
(compressed-tensors pack-quantized int4) is wired into the preset via
`--quant 4bit` / `QWEN3_OMNI_QUANT=4bit`. With our current pinned stack
(transformers 5.9, compressed-tensors 0.11/0.12) the checkpoint loads
without crashing but emits **garbled tokens** — the runtime
dequantization math doesn't agree with how the ckpt was packed
(transformers 4.57.0.dev0 + compressed-tensors 0.11.0).

For 24-GB-class cards today, use the sibling `flashlabs-chroma` preset
instead. We'll re-enable Qwen3-Omni 4-bit when the upstream
compressed-tensors / transformers integration stabilises for this
checkpoint, or when an alternative quant (e.g. a clean AWQ or GPTQ
re-pack) is published.

## Text-only mode (save ~10 GB VRAM)

```bash
QWEN3_OMNI_TEXT_ONLY=1 cvl run qwen3-omni predict
# or:
bash examples/generative/audio/qwen3_omni/predict.sh --text-only
```

Calls `model.disable_talker()` so the Talker stage is dropped entirely
(not just `return_audio=False`). Useful when you only want the text
response.

## Sample

**Input** — reuses the sibling `flashlabs_chroma` preset's dialogue prompt
(~1.5 s, auto-downloaded):

[`flashlabs_chroma/hi_who_are_you.wav`](https://huggingface.co/datasets/zzsi/cvl/resolve/main/flashlabs_chroma/hi_who_are_you.wav)
— a short "Hi, who are you?" clip synthesized with Piper.

**Output** — a printed text reply on stdout (the Thinker's response) plus
`qwen3_omni_response.wav` (24 kHz mono PCM) in the cwd. The text reply is
printed between two `===` border lines for easy parsing from logs.

## Other modalities (image / video)

The wrapper currently only wires up the **audio-in** path. Qwen3-Omni
itself accepts arbitrary mixes of audio / image / video / text in the
conversation; to add an image turn you'd extend `build_conversation()` in
`predict.py` to append an `{"type": "image", "image": "..."}` entry to the
user content list (and toggle `use_audio_in_video=True` if you want a
video's audio track included). See upstream's `web_demo.py` for the full
multimodal recipe.

## Notes

- Backend: **HF transformers** (`Qwen3OmniMoeForConditionalGeneration` +
  `Qwen3OmniMoeProcessor`) with `qwen_omni_utils.process_mm_info` for
  preprocessing. The upstream README also recommends `vllm-project/vllm-omni`
  for production latency; that is a separate binary and is **not** what
  this preset runs.
- The default AWQ-4bit checkpoint
  ([`cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit`](https://huggingface.co/cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit))
  uses **compressed-tensors pack-quantized int4** (group=32), not classic
  AutoAWQ — `compressed-tensors` is the backend dep, not `autoawq`.
- FlashAttention-2 is **opt-in** (`--attn-impl flash_attention_2` or
  `QWEN3_OMNI_ATTN=flash_attention_2`) and requires installing
  `flash-attn` separately; we default to SDPA to keep the image lean.
- This preset is the file-in / text + file-out alternative to the
  upstream gradio `web_demo.py`. For mic / camera UX, run upstream's demo
  on your host (it uses gradio + browser, not pyaudio, so it does play
  reasonably with the host's mic).
