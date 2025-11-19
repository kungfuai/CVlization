# PaliGemma2 (Detection) Finetune

Scripted finetune of `PaliGemma2` on object detection (converted from the Roboflow Colab).

## VRAM

Notebook requires **A100 with 40GB VRAM** to train (per notebook author).

## Quick Start

```bash
cd examples/perception/vision_language/paligemma2_detection

# Build image (installs training deps)
bash build.sh

# Run training (expects Roboflow JSONL format)
# Adjust paths for your dataset
bash train.sh \
  --train-jsonl /data/dataset/_annotations.train.jsonl \
  --val-jsonl /data/dataset/_annotations.valid.jsonl \
  --image-dir /data/dataset
```

The JSONL should follow Roboflow paligemma format with `prefix` and `suffix` fields.

## Notes

- Vision encoder and projector are frozen, matching the Colab recipe.
- HF cache is mounted into the container to avoid re-downloading weights.
- Dependencies pinned to match the Colab (`transformers==4.47.0`, `roboflow`, `supervision`, `peft`, `bitsandbytes`).
