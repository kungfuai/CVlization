# Kosmos2.5 Grounded OCR Finetune

Script version of the Hugging Face notebook “Fine-tuning KOSMOS2.5 on Grounded OCR”. Trains `microsoft/kosmos-2.5` on the DocLayNet subset with prompt `<ocr>` predictions.

## Quick Start

```bash
cd examples/perception/vision_language/kosmos2_grounded_ocr

# Build CUDA image with required deps
bash build.sh

# Train (downloads merve/doclaynet-small by default)
bash train.sh \
  --output-dir outputs/kosmos2_5_grounded
```

Optional flags:
- `--dataset-name`: Hugging Face dataset id (default `merve/doclaynet-small`)
- `--dataset-split`: Split to use before re-splitting (default `test`)
- `--train-size`: Portion to keep for train (default 0.9)
- `--model-id`: Base checkpoint (`microsoft/kosmos-2.5` by default)
- `--push-to-hub`: Enable HF upload (requires token)

## Notes

- VRAM: notebook expects at least a single A100 class GPU (bf16). Training uses small batch size (1) with grad accumulation 4.
- Training data stays on host; mount directories or set `HF_DATASETS_CACHE` to reuse downloads.
- Trackio logging is installed; training reports to Trackio by default. Pass `--report-to none` (or another list) to disable/change reporting.
