# LightOnOCR-1B (vLLM)

Dockerized example for LightOnOCR-1B-1025 using vLLM’s OpenAI-compatible server plus a lightweight HTTP client.

## Quick Start

```bash
cd examples/perception/vision_language/lighton_ocr

# Build image
bash build.sh

# Run a request (OCR by default, loads model locally in vLLM)
bash predict.sh --image ../test_images/sample.jpg

# (Optional) run a shared server instead of local inference
bash serve.sh   # starts vLLM OpenAI server on :8000
```

## Presets (via CVL CLI)

```bash
cvl run lighton_ocr build
cvl run lighton_ocr serve      # starts the vLLM server
cvl run lighton_ocr predict    # local inference
cvl run lighton_ocr test       # tiny smoke test (OCR on sample.jpg)
```

## Options

- `--image`: Path/URL to an image. Defaults to `../test_images/sample.jpg`.
- `--pdf`: Path/URL to a PDF (first page rendered to PNG, saved in `/tmp`).
- `--page`: PDF page index (0-based, default: 0).
- `--prompt`: Custom prompt (default: “Extract all text from this image.”).
- `--max-new-tokens`: Generation cap (default: 4096).
- `--model-id`: Model to load (default: `lightonai/LightOnOCR-1B-1025`).
- `--tp-size`: Tensor parallel shards (default: 1).
- `--max-model-len`: Override max sequence length (default: model config).

Environment vars:
- `LIGHTON_OCR_MODEL_ID`: Override model name.
- `CVL_IMAGE`: Override Docker image tag (default: `lighton-ocr`).
- `LIGHTON_OCR_EXTRA_SERVE_ARGS`: Extra flags forwarded to `vllm serve` (e.g., `--tensor-parallel-size 2`).

## Notes

- Hugging Face cache is mounted into the container to avoid re-downloading weights across runs.
- PDF pages are rendered to PNG at ~200 DPI (longest side capped to 1540 px, matching model config).
- The example trusts remote code when loading the model (`--trust-remote-code`).
- Downloads performed by the client (e.g., `--pdf` or image URLs) are saved under `/tmp`.
