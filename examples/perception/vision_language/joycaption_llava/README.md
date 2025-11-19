# Llama JoyCaption (LLaVA) Dockerized Example

Inference wrapper for `fancyfeast/llama-joycaption-beta-one-hf-llava` using standard Hugging Face `transformers` (no vLLM needed).

## Quick Start

```bash
cd examples/perception/vision_language/joycaption_llava

# Build image
bash build.sh

# Run a caption request (defaults to sample.jpg)
bash predict.sh --image ../test_images/sample.jpg
```

## Presets (via CVL CLI)

```bash
cvl run joycaption-llava build
cvl run joycaption-llava predict    # caption sample.jpg by default
cvl run joycaption-llava test       # smoke test
```

## Options

- `--image`: Path/URL to an image (downloads to `/tmp` if URL). Default: `../test_images/sample.jpg`.
- `--prompt`: Custom prompt text. Default: `Write a long descriptive caption for this image in a formal tone.`
- `--max-new-tokens`: Generation cap (default: 256).
- `--temperature`: Sampling temperature (default: 0.6).
- `--top-p`: Nucleus sampling (default: 0.9).
- `--output`: Output path (default: `outputs/joycaption.txt`).
- `--format`: `txt` or `json` output.

Environment vars:
- `JOYCAPTION_MODEL_ID`: Override model name (default: `fancyfeast/llama-joycaption-beta-one-hf-llava`).
- `CVL_IMAGE`: Override Docker image tag (default: `joycaption-llava`).

## Notes

- Uses `transformers` `AutoProcessor` + `LlavaForConditionalGeneration` in bf16, device auto-mapped to CUDA if available.
- Hugging Face cache is mounted to share weights across runs.
- Downloads (image URLs) are saved under `/tmp`.
