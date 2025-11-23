# RT-DETR Object Detection (Inference)

Single-image object detection using RT-DETRv2 TorchHub by default (fast `rtdetrv2_r18vd`) and a small camera sample (`examples/ref1.png`, symlink to the shared camera image in the repo). A Transformers backend is available if you have the CUDA dev toolkit installed for the custom deformable attention op.

## Quickstart

```bash
# Build
cvl run rtdetr build

# Predict (TorchHub backend, default sample)
cvl run rtdetr predict

# Predict with Transformers backend and a custom model (requires CUDA_HOME + build tools)
cvl run rtdetr predict -- --backend transformers --model-id PekingU/rtdetr_r34vd --image examples/ref1.png
```

Outputs are written to `outputs/rtdetr/prediction.png` and `outputs/rtdetr/predictions.json`.

## Notes
- Base image already includes PyTorch with CUDA; no extra torch install is needed.
- TorchHub downloads weights from `lyuwenyu/RT-DETR`; first call may take ~120 MB for weights.
- Transformers backend uses HF checkpoints but needs the CUDA dev toolkit to compile multi-scale deformable attention (set `CUDA_HOME`). If unavailable, stick with TorchHub.
- Use `--input-size` (default 640) to match training resolution for TorchHub models.
