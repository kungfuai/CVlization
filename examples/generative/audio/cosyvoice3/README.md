# Fun-CosyVoice3-0.5B

Text-to-speech model with zero-shot voice cloning. Supports Chinese, English, and 7 other languages.

## Usage

```bash
# Build the Docker image
./build.sh

# Generate speech (model downloads on first run, ~10GB)
./predict.sh --text "Hello, world!"

# Specify output path
./predict.sh --text "Hello, world!" --output my_speech.wav
```

## Caching

Model weights are downloaded lazily on first run and cached to your host machine:
- **ModelScope cache**: `~/.cache/modelscope` (mounted into container)
- **HuggingFace cache**: `~/.cache/huggingface` (mounted into container)

Subsequent runs reuse cached weights. The cache is shared across all CVlization examples.

## Inference Modes

**Zero-shot** (default): Clone a voice from reference audio.
```bash
./predict.sh --text "Text to say" --prompt-wav reference.wav --prompt-text "What was said in reference"
```

**Cross-lingual**: Use a voice for a different language.
```bash
./predict.sh --text "<|en|>Hello" --mode cross_lingual --prompt-wav chinese_speaker.wav
```

**Instruct**: Control dialect, speed, or style via text instructions.
```bash
./predict.sh --text "你好" --mode instruct --instruct "请用广东话说"
```

## Options

| Flag | Description |
|------|-------------|
| `--text` | Text to synthesize |
| `--input` | Text file path |
| `--output` | Output WAV path (default: outputs/speech.wav) |
| `--mode` | zero_shot, cross_lingual, or instruct |
| `--prompt-wav` | Reference audio for voice cloning |
| `--prompt-text` | Transcript of reference audio |
| `--instruct` | Instruction text (for instruct mode) |
| `--speed` | Speed multiplier (default: 1.0) |
| `--fp16` | Use half precision (faster, requires GPU) |

## Requirements

- NVIDIA GPU with ~12GB VRAM
- ~20GB disk for model weights (cached)

## Notes

- A sample reference audio is bundled in the Docker image for testing
- Output sample rate is 24kHz
- First run downloads model (~10GB), subsequent runs are faster

## Links

- [GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [Paper](https://arxiv.org/abs/2505.17589)
- [Model](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
