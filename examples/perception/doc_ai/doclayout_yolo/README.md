# DocLayout-YOLO

Document layout detection using [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), a YOLOv10-based model for detecting layout elements in document images.

## Overview

DocLayout-YOLO detects 11 layout element classes:
- Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title

The model is trained on DocSynth-300K synthetic dataset and fine-tuned on various document benchmarks.

## Quick Start

```bash
# Build the Docker image
./build.sh

# Run inference on sample image
./predict.sh

# Run on custom image
./predict.sh --image /path/to/document.jpg
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | sample image | Path to input document image |
| `--output` | `outputs/detections.json` | Where to save detection results |
| `--output-image` | `outputs/annotated.jpg` | Where to save annotated image |
| `--conf` | `0.2` | Confidence threshold |
| `--imgsz` | `1024` | Input image size |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |
| `--no-annotated-image` | false | Skip saving annotated image |

## Output Format

The output JSON contains:
```json
{
  "image": "/path/to/input.jpg",
  "model": "juliozhao/DocLayout-YOLO-DocStructBench",
  "confidence_threshold": 0.2,
  "image_size": 1024,
  "num_detections": 15,
  "detections": [
    {
      "class": "Text",
      "class_id": 9,
      "confidence": 0.95,
      "bbox": {"x1": 100.5, "y1": 200.3, "x2": 500.2, "y2": 250.8}
    }
  ]
}
```

## Resources

- **Model**: [juliozhao/DocLayout-YOLO-DocStructBench](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench)
- **Paper**: [DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception](https://arxiv.org/abs/2410.12628)
- **GitHub**: [opendatalab/DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

## Requirements

- NVIDIA GPU with ~4GB VRAM
- Docker with NVIDIA runtime
