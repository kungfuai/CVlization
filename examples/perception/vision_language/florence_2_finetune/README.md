# Florence-2 DocVQA Finetune

Scripted port of the HF notebook “Fine-tuning Florence-2 on DocVQA”. This trains `microsoft/Florence-2-base-ft` (revision `refs/pr/6`) on the `HuggingFaceM4/DocumentVQA` dataset.

## VRAM

Plan for at least one high-memory GPU (the notebook uses ~24 GB). Training uses batch size 6 with 2 epochs by default; adjust for your hardware.

## Quick Start

```bash
cd examples/perception/vision_language/florence_2_finetune

# Build the CUDA image with flash-attn, timm, einops, etc.
bash build.sh

# Train on DocumentVQA (downloads automatically)
bash train.sh \
  --output-dir outputs/florence2_docvqa
```

Key arguments (see `python train.py -h`):

- `--model-id`: Base checkpoint (default `microsoft/Florence-2-base-ft`).
- `--revision`: Git revision (`refs/pr/6` by default).
- `--epochs`, `--batch-size`, `--lr`, etc.
- `--freeze-vision`: Freeze the vision tower (default enabled).
- `--push-to-hub`: Push model/processor to your HF repo (requires token).

## Notes

- Training loop mirrors the Colab: manual PyTorch loop with AdamW + linear scheduler.
- Dataloaders build prompts `<DocVQA>{question}` and use the first answer as target.
- Checkpoints are stored per epoch under `checkpoints/` and the final model lives in `--output-dir`.
- FlashAttention requires `ninja-build`; the Dockerfile installs it to ensure pip build success.

## References

- smol-vision notebook: https://huggingface.co/merve/smol-vision/blob/main/Fine_tune_Florence_2.ipynb
