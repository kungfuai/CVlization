## docTR - Document Text Recognition

End-to-end OCR using [docTR](https://github.com/mindee/doctr), combining text detection and recognition in a single pipeline.

### Features

- **Text Detection**: Multiple architectures (DB, LinkNet)
- **Text Recognition**: CRNN, MASTER, SAR models
- **Configurable**: Choose speed vs accuracy tradeoffs
- **Multiple Formats**: Text and JSON output

### Prerequisites

1. NVIDIA GPU (recommended for faster inference)
2. Docker with NVIDIA runtime support

### Quickstart

1. Build the Docker image:
```bash
bash examples/perception/doc_ai/doctr/build.sh
```

2. Run inference:
```bash
bash examples/perception/doc_ai/doctr/predict.sh --image examples/sample.jpg
```

### Usage

#### Basic Inference
```bash
bash examples/perception/doc_ai/doctr/predict.sh --image document.jpg
```

#### JSON Output
```bash
bash examples/perception/doc_ai/doctr/predict.sh --image scan.jpg --format json
```

#### Custom Architectures
```bash
# Fast inference with lightweight models
bash examples/perception/doc_ai/doctr/predict.sh \
    --image receipt.png \
    --det-arch db_mobilenet_v3_large \
    --reco-arch crnn_mobilenet_v3_small

# High accuracy with MASTER
bash examples/perception/doc_ai/doctr/predict.sh \
    --image complex_doc.jpg \
    --reco-arch master
```

### Command-Line Options

- `--image`: Path to input image or URL (default: `examples/sample.jpg`)
- `--format`: Output format - `txt` or `json` (default: `txt`)
- `--det-arch`: Detection architecture (default: `db_resnet50`)
  - `db_resnet50` - ResNet50 backbone
  - `db_mobilenet_v3_large` - MobileNetV3 (faster)
  - `linknet_resnet18` - LinkNet with ResNet18
- `--reco-arch`: Recognition architecture (default: `crnn_vgg16_bn`)
  - `crnn_vgg16_bn` - CRNN with VGG16
  - `crnn_mobilenet_v3_small` - CRNN with MobileNetV3 (faster)
  - `master` - MASTER architecture (higher accuracy)
  - `sar_resnet31` - SAR with ResNet31
- `--no-gpu`: Disable GPU acceleration

### Model Details

docTR uses a two-stage approach:
1. **Detection**: Localizes text regions using differentiable binarization (DB)
2. **Recognition**: Extracts text using CRNN, MASTER, or SAR architectures

All models are pretrained and optimized for production use.

### Directory Structure

```
examples/perception/doc_ai/doctr/
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── predict.py          # Main inference script
├── build.sh            # Build Docker image
├── predict.sh          # Run inference wrapper
├── example.yaml        # CVL configuration
├── README.md           # This file
├── .gitignore          # Git ignore rules
├── examples/           # Sample images
└── outputs/            # Results saved here
```

### Output Location

By default, outputs are saved to:
- `examples/perception/doc_ai/doctr/outputs/`

### References

- [docTR GitHub](https://github.com/mindee/doctr)
- [docTR Documentation](https://mindee.github.io/doctr/)

### License

This example follows the Apache 2.0 license of the docTR library.
