# VibeVoice-Realtime-0.5B

Real-time text-to-speech inference using Microsoft's VibeVoice-Realtime-0.5B model.

## Model Overview

- **Model**: [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
- **Parameters**: 0.5B (based on Qwen2.5-0.5B)
- **Latency**: ~300ms to first audio
- **Output**: 24kHz audio
- **Max duration**: ~10 minutes of speech
- **Language**: English only

## Requirements

- NVIDIA GPU with 8GB+ VRAM (T4 or better recommended)
- Docker with NVIDIA Container Toolkit
- ~15GB disk space for model and container

## Quick Start

```bash
# Build the Docker image
./build.sh

# Generate speech from text
./predict.sh --text "Hello world, this is a test."

# Generate speech from file
./predict.sh --input sample.txt --output outputs/sample.wav

# Run smoke test
./test.sh
```

## Usage

### Direct Text Input

```bash
./predict.sh --text "Your text here" --output outputs/speech.wav
```

### From Text File

```bash
./predict.sh --input sample.txt --output outputs/speech.wav
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--text` | Text to synthesize (direct) | - |
| `--input` | Path to text file | - |
| `--output` | Output WAV file path | `outputs/speech.wav` |
| `--speaker` | Speaker voice name (see below) | `Carter` |
| `--cfg-scale` | Classifier-free guidance scale | `1.5` |
| `--device` | Device (cuda/cpu) | auto-detect |
| `--no-flash-attn` | Disable flash attention | - |
| `--verbose` | Enable verbose logging | - |

## Available Speakers

**English voices:**
- `Carter` (man) - default
- `Davis` (man)
- `Frank` (man)
- `Mike` (man)
- `Emma` (woman)
- `Grace` (woman)

**Other languages (experimental, unsupported):**
- German: `Spk0` (man), `Spk1` (woman)
- French: `Spk0` (man), `Spk1` (woman)
- Italian: `Spk0` (woman), `Spk1` (man)
- Japanese: `Spk0` (man), `Spk1` (woman)
- Korean: `Spk0` (woman), `Spk1` (man)
- Dutch: `Spk0` (man), `Spk1` (woman)
- Polish: `Spk0` (man), `Spk1` (woman)
- Portuguese: `Spk0` (woman), `Spk1` (man)
- Spanish: `Spk0` (woman), `Spk1` (man)
- Indian English: `Samuel` (man)

## Limitations

- English only (other languages may produce unpredictable results)
- Single speaker per generation
- Very short inputs (<4 words) may produce degraded output
- Does not support: music, background audio, code, formulas, special symbols
- Model includes audible AI disclosure disclaimer in output

## References

- Model Card: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B
- GitHub: https://github.com/microsoft/VibeVoice
- Technical Report: https://arxiv.org/abs/2508.19205
