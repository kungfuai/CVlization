# Florence-2-Large - Microsoft's Unified Vision Foundation Model

Florence-2-Large is a compact vision language model from Microsoft Research with 0.77B parameters. It provides a unified, prompt-based interface for multiple vision tasks including captioning, OCR, object detection, and segmentation.

## Model Information

- **Model**: `microsoft/Florence-2-large`
- **Size**: 0.77B parameters (770M)
- **VRAM**: ~2GB
- **License**: MIT
- **Paper**: [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242)

## Features

### Task Capabilities

Florence-2 supports multiple vision tasks through simple text prompts:

| Task | Prompt | Description |
|------|--------|-------------|
| **Caption** | `<CAPTION>` | Basic image captioning |
| **Detailed Caption** | `<DETAILED_CAPTION>` | More comprehensive descriptions |
| **Extended Caption** | `<MORE_DETAILED_CAPTION>` | Very detailed descriptions |
| **OCR** | `<OCR>` | Text extraction |
| **OCR with Regions** | `<OCR_WITH_REGION>` | Text extraction with bounding boxes |
| **Object Detection** | `<OD>` | Detect objects with bounding boxes |
| **Dense Region Caption** | `<DENSE_REGION_CAPTION>` | Caption multiple regions |
| **Region Proposal** | `<REGION_PROPOSAL>` | Suggest regions of interest |

## Quick Start

### 1. Build the Docker Image

```bash
bash build.sh
```

This creates a ~10GB Docker image with all dependencies.

### 2. Run Inference

#### Image Captioning

```bash
bash predict.sh --image test_images/sample.jpg --task caption
```

#### Detailed Captioning

```bash
bash predict.sh --image photo.jpg --task detailed_caption
```

#### OCR (Text Extraction)

```bash
bash predict.sh --image document.jpg --task ocr
```

#### Object Detection

```bash
bash predict.sh --image scene.jpg --task object_detection
```

#### OCR with Bounding Boxes

```bash
bash predict.sh --image receipt.jpg --task ocr_with_region --format json
```

### 3. Run Tests

```bash
bash test.sh
```

## Usage

### Basic Usage

```bash
bash predict.sh --image <path> --task <task_name>
```

### All Options

```bash
bash predict.sh \
  --image <path_or_url> \
  --task <task_name> \
  --output <output_path> \
  --format <txt|json> \
  --device <cuda|mps|cpu>
```

### Available Tasks

- `caption` - Basic image caption
- `detailed_caption` - Detailed description
- `more_detailed_caption` - Extended description
- `ocr` - Extract text
- `ocr_with_region` - Extract text with locations
- `object_detection` - Detect objects
- `dense_region_caption` - Caption image regions
- `region_proposal` - Suggest regions

## Output Formats

### Text Output (default)

Simple text results saved to file:

```bash
bash predict.sh --image photo.jpg --task caption --output result.txt
```

### JSON Output

Structured output with metadata and bounding boxes:

```bash
bash predict.sh --image doc.jpg --task object_detection --output result.json --format json
```

Example JSON output for object detection:

```json
{
  "labels": ["person", "car", "tree"],
  "bboxes": [[100, 150, 200, 300], [300, 200, 450, 400], [500, 100, 600, 350]]
}
```

## Test Images

This example uses shared test images from `../test_images/` to avoid duplicating image files across multiple VLM examples. See `../test_images/README.md` for details.

## Performance

- **VRAM Usage**: ~2GB
- **Model Size**: 8.3GB on disk
- **Speed**: Fast inference on GPU (<1s per image)
- **Accuracy**: Strong zero-shot performance across vision tasks

## Architecture

Florence-2 uses a unified sequence-to-sequence architecture with:
- Vision encoder for image understanding
- Language decoder for text generation
- Task-specific prompts for different capabilities
- Support for both text and structured outputs (bboxes, labels)

## Notes

- The model uses `trust_remote_code=True` for loading
- GPU recommended for fast inference (works on CPU but slower)
- Use float16 on CUDA for memory efficiency
- Some tasks return structured data (bounding boxes, labels)
- OCR and object detection work best with clear, high-quality images

## Citation

```bibtex
@article{xiao2023florence,
  title={Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks},
  author={Xiao, Bin and Wu, Haiping and Xu, Weijian and Dai, Xiyang and Hu, Houdong and Lu, Yumao and Zeng, Michael and Liu, Ce and Yuan, Lu},
  journal={arXiv preprint arXiv:2311.06242},
  year={2023}
}
```

## License

MIT License - See model card at https://huggingface.co/microsoft/Florence-2-large
