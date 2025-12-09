# Gemma-3 Vision Inference

Vision-language inference using Google's Gemma-3 multimodal model.

## Quick Start

```bash
# Build
./build.sh

# Test
./test.sh

# Run on your image
./predict.sh --image path/to/image.jpg --task describe
```

Or via cvl CLI:
```bash
cvl run gemma3-vision build
cvl run gemma3-vision test
cvl run gemma3-vision predict --image path/to/image.jpg
```

## Tasks

- `describe` - Describe the image in detail
- `ocr` - Extract text from the image
- `caption` - Short caption
- `analyze` - Analyze what's happening
- `query` - Custom prompt (use with `--prompt "your question"`)

## Options

```bash
./predict.sh --image image.jpg \
    --task query \
    --prompt "What color is the car?" \
    --max-tokens 512 \
    --temperature 1.0 \
    --output result.json \
    --format json
```

### Model Selection

Default is `google/gemma-3-4b-it`. For larger models:

```bash
./predict.sh --image image.jpg --model-id google/gemma-3-12b-it
```

| Model | VRAM (4-bit) |
|-------|--------------|
| gemma-3-4b-it | ~8GB |
| gemma-3-12b-it | ~12GB |
| gemma-3-27b-it | ~20GB |

### Memory Options

```bash
# Disable 4-bit quantization (more VRAM, slightly better quality)
./predict.sh --image image.jpg --no-4bit

# Disable pan & scan for high-res images
./predict.sh --image image.jpg --no-pan-scan
```

## Requirements

- NVIDIA GPU with 8GB+ VRAM
- Docker with NVIDIA Container Toolkit
- HuggingFace account with Gemma license accepted

Accept the license at: https://huggingface.co/google/gemma-3-4b-it

## Output

Results are saved to `outputs/result.txt` by default. Use `--format json` for structured output with metadata.
