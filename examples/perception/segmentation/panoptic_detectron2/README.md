# Panoptic Segmentation (Detectron2)

This example runs Detectron2's PanopticFPN inference pipeline and saves a visualized panoptic output image.

## Quickstart

```bash
cvl run panoptic-segmentation-detectron2 build
cvl run panoptic-segmentation-detectron2 predict
```

Optional custom input:

```bash
cvl run panoptic-segmentation-detectron2 predict -- --input /path/to/image.jpg --output outputs/panoptic.jpg
```

Notes:
- Uses COCO-pretrained `panoptic_fpn_R_50_3x` from Detectron2 model zoo.
- If no input is provided, a sample image is downloaded.
