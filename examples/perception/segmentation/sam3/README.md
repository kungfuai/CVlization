# SAM3 Inference (Text-Prompted Segmentation)

Lightweight, dockerized SAM3 inference. Builds a runtime-only image, loads the gated `facebook/sam3` checkpoint (or a local fine-tuned checkpoint), and runs text-prompted segmentation on a single image.

## Quickstart

```bash
cd examples/perception/segmentation/sam3

# 1) Build inference image (runtime base, torch already in image)
./build.sh

# 2) Run prediction (GPU if available, CPU otherwise)
# Defaults: image=examples/sample.jpg (invoice), prompt="text"
./predict.sh

# Custom inputs
./predict.sh --image examples/sample.jpg --text "person" --output outputs/sam3/person.png
```

- If downloading the HF checkpoint for the first time, set `HF_TOKEN` in `.env` at repo root or export it before running.
- Use `--checkpoint /path/to/finetuned_checkpoint.pt` to test your own weights.

## Script reference

- `predict.py`: transformers path (default); falls back to the cloned SAM3 repo if `Sam3Model` is unavailable. Switch with `--model_loader transformers|repo`, tweak thresholds with `--score-threshold` / `--mask-threshold`.
- `predict_with_sam3_repo.py`: legacy path using the cloned SAM3 repo directly.
- `predict.sh`: docker wrapper; mounts Hugging Face cache, reads `HF_TOKEN`, and runs with GPU if `CUDA_VISIBLE_DEVICES` is set.
- `Dockerfile`: runtime PyTorch base + minimal inference deps; clones SAM3 at commit `84cc43b` for reproducibility (needed by the legacy script).

Note: we pin transformers to a specific commit for SAM3 support. Once an official release newer than 4.57.1 includes SAM3, switch the requirement to that release instead of the git hash.

## Options (predict.py)

- `--image`: input image path (inside the container; use `predict.sh` to mount current example dir).
- `--text`: text prompt for the concept(s) to segment.
- `--checkpoint`: HF model id or local checkpoint path (default `facebook/sam3`).
- `--output`: overlay PNG path (default `outputs/sam3/prediction.png`).
- `--mask-threshold`: binarization threshold for masks (default 0.5).

## Notes

- GPU is recommended; CPU works but will be slow for an 848M-parameter model.
- HF checkpoint is gated; ensure `HF_TOKEN` has access. Cache is mounted at `/root/.cache/huggingface`.
- The example mounts only this directory. Place your image here (or adjust `predict.sh` mounts) so the container can read it.
