# YOLOv13 Object Detection (Inference)

Single-image detection using the ultralytics YOLOv13 models. Defaults to the nano weights (`yolov13n.pt`) and the built-in bus sample from the ultralytics assets.

## Quickstart

```bash
# Build
cvl run yolov13 build

# Predict (nano weights, built-in bus sample)
cvl run yolov13 predict

# Predict with custom weights and image
cvl run yolov13 predict -- --weights yolov13s.pt --image /mnt/cvl/workspace/path/to/image.jpg
```

Outputs are written to `outputs/yolov13/prediction.png` and `outputs/yolov13/predictions.json`.

## Notes
- Base image already includes PyTorch with CUDA; no extra torch install needed.
- We install yolov13 from GitHub at commit `7328994` plus `opencv-python-headless` for image IO.
- The first run will download the specified weights if not cached.
- Default image uses the ultralytics packaged bus sample; pass `--image` to use your own (repo mount at `/mnt/cvl/workspace` via `predict.sh`).
