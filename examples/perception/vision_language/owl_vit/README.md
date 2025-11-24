## OWL-ViT: Open-Vocabulary Object Detection

Open-vocabulary object detection using OWL-ViT/OWLv2. Detects objects based on text queries without training on specific categories.

### What is OWL-ViT?

OWL-ViT is an object detector developed by Google Research that can find objects described in natural language. Unlike traditional detectors trained on fixed classes (e.g., 80 COCO classes), it uses CLIP as a backbone to match image regions with text descriptions.

Key characteristics:
- Text-conditioned detection using CLIP embeddings
- No retraining needed for new object categories
- Zero-shot detection on arbitrary categories
- Two versions: v1 (2022) and v2 (2023, better performance)

### Prerequisites

- NVIDIA GPU with 8GB+ VRAM
- Docker with NVIDIA runtime
- HuggingFace account (optional)

### Quickstart

Build and run:
```bash
bash examples/perception/vision_language/owl_vit/build.sh
bash examples/perception/vision_language/owl_vit/predict.sh --image path/to/image.jpg
```

### Basic Usage

Detect objects with text queries:
```bash
bash examples/perception/vision_language/owl_vit/predict.sh \
    --image photo.jpg \
    --queries "a photo of a cat" "a photo of a dog"
```

Adjust confidence threshold:
```bash
bash examples/perception/vision_language/owl_vit/predict.sh \
    --image photo.jpg \
    --queries "a person" \
    --threshold 0.2
```

Save as JSON:
```bash
bash examples/perception/vision_language/owl_vit/predict.sh \
    --image photo.jpg \
    --format json
```

### Available Models

| Model | Size | LVIS APr | Notes |
|-------|------|----------|-------|
| owlvit-base-patch32 | 583 MB | 16.9% | Fastest |
| owlvit-base-patch16 | 581 MB | 17.1% | Balanced |
| owlvit-large-patch14 | 1.6 GB | 31.2% | Best v1 |
| owlv2-base-patch16 | 590 MB | 36.2% | Default |
| owlv2-large-patch14 | 1.7 GB | 44.0% | Best overall |

Use a specific model:
```bash
bash examples/perception/vision_language/owl_vit/predict.sh \
    --model owlv2-large-patch14 \
    --image photo.jpg
```

### Command Options

```
--image PATH          Input image path
--output PATH         Output file path
--format FORMAT       Output: image, json, or both (default: both)
--model MODEL         Model variant (default: owlv2-base-patch16)
--queries TEXT...     Text descriptions of objects to detect
--threshold FLOAT     Confidence threshold 0.0-1.0 (default: 0.1)
--device DEVICE       Device: auto, cuda, cpu, mps (default: auto)
--no-labels          Hide labels on output image
```

### Performance

Tested on NVIDIA A10 (24GB VRAM):
- OWLv2 Base/16: ~5s load, ~0.5s inference, ~2GB VRAM
- OWLv2 Large/14: ~8s load, ~1.2s inference, ~4GB VRAM

### Query Tips

Queries should be specific and natural:
- Good: "a photo of a red car", "a person wearing sunglasses"
- Avoid: "something", "object", "thing"

Including "a photo of" often improves results due to CLIP's training.

### Limitations

- Lower accuracy than specialized detectors on their trained classes
- Results depend on query phrasing
- May miss small objects (common for ViT-based models)
- Slower than YOLO-style detectors

### Comparison

| Model | Categories | Flexibility | Retraining |
|-------|------------|-------------|------------|
| OWL-ViT | Any text | High | None |
| YOLO | Fixed 80 | None | Required |
| Faster R-CNN | Fixed | None | Required |

### Troubleshooting

No detections found:
- Lower threshold: `--threshold 0.05`
- Improve queries: be more specific
- Try larger model: `--model owlv2-large-patch14`

Out of memory:
- Use base model: `--model owlv2-base-patch16`
- Try CPU: `--device cpu` (slower)

### Directory Structure

```
owl_vit/
├── Dockerfile          # PyTorch 2.9.0 runtime
├── requirements.txt    # transformers, pillow, scipy
├── predict.py         # Inference script
├── build.sh           # Build image
├── predict.sh         # Run container
├── example.yaml       # Metadata
├── README.md          # This file
└── examples/          # Sample images (symlink)
```

### References

- Paper (v1): [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230) (ECCV 2022)
- Paper (v2): [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683) (NeurIPS 2023)
- Code: [Google Research Scenic](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
- Models: [HuggingFace](https://huggingface.co/models?search=google/owlv)

### Citation

```bibtex
@inproceedings{minderer2022simple,
  title={Simple Open-Vocabulary Object Detection with Vision Transformers},
  author={Minderer, Matthias and Gritsenko, Alexey and Stone, Austin and Neumann, Maxim and Weissenborn, Dirk and Dosovitskiy, Alexey and Mahendran, Aravindh and Arnab, Anurag and Dehghani, Mostafa and Shen, Zhuoran and others},
  booktitle={European Conference on Computer Vision},
  year={2022}
}

@inproceedings{minderer2023scaling,
  title={Scaling Open-Vocabulary Object Detection},
  author={Minderer, Matthias and Gritsenko, Alexey and Houlsby, Neil},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
