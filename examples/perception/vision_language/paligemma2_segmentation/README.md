# PaliGemma2 (Instance Segmentation) Finetune

Scripted finetune of `PaliGemma2` on instance segmentation (converted from the Roboflow Colab).

## VRAM

Notebook requires **A100 with 40GB VRAM** to train (per notebook author).

## Quick Start

```bash
cd examples/perception/vision_language/paligemma2_segmentation

# Build image (installs training deps)
bash build.sh

# Convert COCO masks to Paligemma format and train
bash train.sh \
  --dataset-dir /data/fashion-assistant-segmentation \
  --output-dir outputs/paligemma2_seg
```

## Notes

- Converts COCO-style segmentation annotations to the codebook format used in the Colab, then trains with a simple Trainer loop.
- Vision encoder/projector are frozen (per notebook).
- Hugging Face cache is mounted into the container to avoid re-downloading weights.
- Dependencies pinned to match the Colab (`transformers==4.47.0`, `roboflow`, `supervision`, `peft`, `bitsandbytes`, `tensorflow`, `big_vision` code).
