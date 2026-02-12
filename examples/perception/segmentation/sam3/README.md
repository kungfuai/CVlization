# SAM3 Inference (Text-Prompted Segmentation)

Lightweight, dockerized SAM3 inference. Builds a runtime-only image, loads the gated `facebook/sam3` checkpoint (or a local fine-tuned checkpoint), and runs text-prompted segmentation on images or videos.

## Quickstart

```bash
cd examples/perception/segmentation/sam3

# 1) Build inference image (runtime base, torch already in image)
./build.sh

# 2) Run prediction (GPU if available, CPU otherwise)
# Defaults: image=examples/sample.jpg (invoice), prompt="text"
./predict.sh

# Custom inputs
./predict.sh --input examples/sample.jpg --text "person" --output outputs/sam3/person.png
```

- If downloading the HF checkpoint for the first time, set `HF_TOKEN` in `.env` at repo root or export it before running.
- Use `--checkpoint /path/to/finetuned_checkpoint.pt` to test your own weights.

## Video Segmentation

SAM3 supports video object tracking: prompt on a single frame and propagate masks across the entire video.

```bash
# Segment and track an object in a video
./predict.sh --input video.mp4 --text "watermark"

# Save per-frame binary mask PNGs (compatible with ProPainter --mask format)
./predict.sh --input video.mp4 --text "watermark" --save-masks

# Control propagation direction
./predict.sh --input video.mp4 --text "person" --propagation-direction both
```

Video mode always uses the native SAM3 repo backend (the HF transformers backend does not support video). Output:

- **Overlay MP4**: `prediction_video.mp4` (default) — colored mask overlay on each frame.
- **Mask PNGs** (with `--save-masks`): `prediction_video_masks/0000.png, 0001.png, ...` — binary white-on-black masks, one per frame. These are directly compatible with ProPainter's `--mask` folder input for video inpainting.

## Script reference

- `predict.py`: unified entry point for image and video segmentation. Auto-detects input type by file extension. Image mode uses transformers (default) or native repo; video mode uses native repo.
- `predict.sh`: docker wrapper; mounts Hugging Face cache, reads `HF_TOKEN`, and runs with GPU if `CUDA_VISIBLE_DEVICES` is set.
- `Dockerfile`: runtime PyTorch base + minimal inference deps; vendors SAM3 source for reproducibility.

Note: we pin transformers to a specific commit for SAM3 support. Once an official release newer than 4.57.1 includes SAM3, switch the requirement to that release instead of the git hash.

## Options (predict.py)

- `--input` / `-i`: input image or video path (auto-detected by extension).
- `--image`: deprecated alias for `--input` (image only).
- `--text`: text prompt for the concept(s) to segment.
- `--checkpoint`: HF model id or local checkpoint path (default `facebook/sam3`).
- `--output`: output path (default `prediction.png` for images, `prediction_video.mp4` for videos).
- `--mask-threshold`: binarization threshold for masks (default 0.5).
- `--model_loader`: `transformers` (default) or `repo` — image mode only.
- `--score-threshold`: instance score threshold (transformers backend, default 0.1).
- `--save-masks`: save per-frame binary mask PNGs (video only).
- `--propagation-direction`: `forward` (default), `backward`, or `both` (video only).

## Notes

- GPU is recommended; CPU works but will be slow for an 848M-parameter model.
- HF checkpoint is gated; ensure `HF_TOKEN` has access. Cache is mounted at `/root/.cache/huggingface`.
- The example mounts only this directory. Place your input here (or adjust `predict.sh` mounts) so the container can read it.
- Video mode requires ffmpeg (included in the Docker image).
