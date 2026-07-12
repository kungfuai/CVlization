# OmniVoice

Massively multilingual zero-shot text-to-speech with voice cloning and voice
design using k2-fsa OmniVoice 0.6B.

## Model Overview

- **Model**: [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
- **Parameters**: 0.6B (Qwen3-0.6B-Base backbone)
- **Sample rate**: 24 kHz
- **Languages**: 646 (English, Chinese, Japanese, Spanish, French, German, and many more)
- **Architecture**: Non-autoregressive diffusion language model
- **Code license**: Apache-2.0
- **Weight license**: CC-BY-NC (due to training-data constraints — not for commercial use)

## Sample

**Voice design** — `--instruct "female, british accent"`:

> *"The quick brown fox jumps over the lazy dog. Testing OmniVoice voice design
> capabilities."*

Output (5.1 s, 24 kHz):
[voice_design_output.wav](https://huggingface.co/datasets/zzsi/cvl/resolve/main/omnivoice/voice_design_output.wav)

**Voice cloning** — same speaker, new text:

> *"Good morning everyone. Today we will discuss the latest developments in
> multilingual speech synthesis technology."*

Output (5.8 s, 24 kHz):
[voice_clone_output.wav](https://huggingface.co/datasets/zzsi/cvl/resolve/main/omnivoice/voice_clone_output.wav)

## What to Expect

- **First run**: downloads ~3.1 GB of model weights; voice cloning without
  `--ref-text` also downloads Whisper (~1.6 GB) for auto-transcription.
  All weights are cached in `~/.cache/huggingface/` afterward.
- **Inference**: generates a single WAV from a text prompt
- **Output**: saved to your working directory as `speech.wav` (24 kHz mono WAV)
- **Runtime**: ~5–15 s per utterance on a modern GPU (32 diffusion steps);
  ~3 s with `--num-step 16`
- **VRAM**: peak ~2.7 GB on RTX PRO 6000 Blackwell (float16)

## Requirements

- NVIDIA GPU with 4 GB+ VRAM (or CPU, much slower)
- Docker with NVIDIA Container Toolkit
- ~10 GB disk space (image + cached model weights)

## Quick Start

```bash
# Build the Docker image
./build.sh

# Voice design — no reference audio needed
./predict.sh --text "Hello world!" --instruct "female, british accent"

# Voice cloning — uses canonical reference audio from zzsi/cvl
./predict.sh --text "Hello world!"

# Voice cloning — with your own reference audio
./predict.sh --text "Hello world!" --ref-audio my_voice.wav --ref-text "Transcript of my voice."

# Run smoke tests
./test.sh
```

## Usage

### Voice Design (no reference audio)

Generate speech with designed voice attributes:

```bash
./predict.sh \
  --text "The quick brown fox jumps over the lazy dog." \
  --instruct "male, low pitch, american accent" \
  --output designed.wav
```

Supported attributes (comma-separated):
- **Gender**: male, female
- **Age**: child, teenager, young adult, middle-aged, elderly
- **Pitch**: very low, low, moderate pitch, high, very high
- **Style**: whisper
- **Accents**: american accent, british accent, australian accent, canadian accent,
  indian accent, japanese accent, korean accent, chinese accent, portuguese accent,
  russian accent

### Voice Cloning (with reference audio)

Clone a voice from a 3–10 second reference clip:

```bash
./predict.sh \
  --text "This text will be spoken in the cloned voice." \
  --ref-audio reference.wav \
  --ref-text "Transcript of the reference audio." \
  --output cloned.wav
```

Providing `--ref-text` avoids downloading the Whisper ASR model (~1.6 GB) for
auto-transcription.

### From Text File

```bash
./predict.sh --input article.txt --instruct "female, high pitch" --output article.wav
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--text` | Text to synthesize (direct) | *(demo sentence)* |
| `--input` | Path to text file | — |
| `--output` | Output WAV file path | `speech.wav` |
| `--ref-audio` | Reference audio for voice cloning | — |
| `--ref-text` | Transcript of reference audio | — |
| `--instruct` | Voice design attributes | — |
| `--num-step` | Diffusion steps (higher = better) | `32` |
| `--speed` | Speech speed factor | `1.0` |
| `--verbose` | Enable verbose logging | — |

## License Notice

The OmniVoice **code** is released under Apache-2.0. The **pretrained model
weights** are licensed CC-BY-NC (non-commercial) because the training data
(Emilia dataset) carries that restriction. If you need weights for commercial
use, you must retrain on a permissively licensed dataset.

## Limitations

- Cross-lingual accent: cloning a voice in a language different from the
  reference may carry the reference language's accent
- Very long inputs may need chunking for stable output
- Voice cloning quality degrades with noisy or very short reference audio
- 24 kHz output (not 44.1/48 kHz studio quality)

## References

- Model card: https://huggingface.co/k2-fsa/OmniVoice
- GitHub: https://github.com/k2-fsa/OmniVoice
- Paper: https://arxiv.org/abs/2604.00688
