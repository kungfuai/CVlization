# VoxCPM1.5

Tokenizer-free text-to-speech with zero-shot voice cloning using OpenBMB's VoxCPM1.5.

## Model Overview

- **Model**: [openbmb/VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5)
- **Parameters**: 0.5B (MiniCPM-4-0.5B backbone)
- **Sample rate**: 44.1kHz
- **Languages**: Chinese + English (bilingual)
- **License**: Apache-2.0

## Features

- High-quality 44.1kHz audio output
- Zero-shot voice cloning from reference audio
- Streaming synthesis support
- Text normalization for numbers/abbreviations
- Optional denoising

## Requirements

- NVIDIA GPU with 8GB+ VRAM
- Docker with NVIDIA Container Toolkit
- ~10GB disk space

## Quick Start

```bash
# Build the Docker image
./build.sh

# Generate speech from text
./predict.sh --text "Hello world, this is VoxCPM."

# Run smoke test
./test.sh
```

## Usage

### Basic Text-to-Speech

```bash
./predict.sh --text "Your text here" --output outputs/speech.wav
```

### Voice Cloning

```bash
./predict.sh \
  --text "Text to speak in cloned voice" \
  --prompt-audio reference.wav \
  --prompt-text "Transcript of reference audio" \
  --output outputs/cloned.wav
```

### From Text File

```bash
./predict.sh --input sample.txt --output outputs/speech.wav
```

### Streaming Mode

```bash
./predict.sh --text "Streaming synthesis" --streaming
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--text` | Text to synthesize (direct) | - |
| `--input` | Path to text file | - |
| `--output` | Output WAV file path | `outputs/speech.wav` |
| `--prompt-audio` | Reference audio for voice cloning | - |
| `--prompt-text` | Transcript of reference audio | - |
| `--cfg-value` | Guidance strength | `2.0` |
| `--timesteps` | Inference timesteps (higher=better) | `10` |
| `--no-normalize` | Disable text normalization | - |
| `--denoise` | Enable denoising (16kHz output) | - |
| `--streaming` | Use streaming synthesis | - |
| `--verbose` | Enable verbose logging | - |

## Parameters

- **cfg-value**: Controls adherence to text. Higher = stricter, lower = more relaxed
- **timesteps**: Quality vs speed tradeoff. 10 is balanced, higher for better quality

## Limitations

- Bilingual only (Chinese/English) - other languages unsupported
- Very long inputs may show instability
- Voice cloning requires clear reference audio
- Denoising limits output to 16kHz

## References

- Model Card: https://huggingface.co/openbmb/VoxCPM1.5
- GitHub: https://github.com/OpenBMB/VoxCPM
- Technical Report: https://arxiv.org/abs/2509.24650
