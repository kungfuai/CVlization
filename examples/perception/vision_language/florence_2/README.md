# Florence-2 (Base & Large)

This unified example exposes both Florence-2 checkpoints from Microsoft Research. Use it for captioning, OCR, detection, and grounding tasks while switching between the compact Base (0.23B, ~1GB VRAM) and the beefier Large (0.77B, ~2GB VRAM) models via a single `--variant` flag.

## Model Information

- **Model Variants**:
  - `--variant base` → `microsoft/Florence-2-base` (~0.23B params, ~1GB VRAM)
  - `--variant large` → `microsoft/Florence-2-large` (~0.77B params, ~2GB VRAM)
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

This creates a Docker image with PyTorch 2.5.1 and required dependencies.

### 2. Run Inference

The commands below default to the Base checkpoint. Append `--variant large` to use the larger model.

#### Image Captioning

```bash
bash predict.sh --variant base --image test_images/sample.jpg --task caption
```

#### Detailed Captioning

```bash
bash predict.sh --variant base --image photo.jpg --task detailed_caption
```

#### OCR (Text Extraction)

```bash
bash predict.sh --variant base --image document.jpg --task ocr
```

#### Object Detection

```bash
bash predict.sh --variant base --image scene.jpg --task object_detection
```

#### OCR with Bounding Boxes

```bash
bash predict.sh --variant base --image receipt.jpg --task ocr_with_region --format json
```

### 3. Run Tests

```bash
bash test.sh
```

## Usage

### Basic Usage

```bash
bash predict.sh --variant <base|large> --image <path> --task <task_name>
```

### All Options

```bash
bash predict.sh \
  --variant <base|large> \
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

Tested on NVIDIA A10 GPU with invoice image (800x600):

- **VRAM Usage**: Base ~1GB, Large ~2GB (float16)
- **Model Download Size**: Base ~0.5GB, Large ~1.5GB
- **Tasks Verified**: Caption + OCR success on both variants (Large provides slightly richer descriptions)
- **Output Format**: Returns text or structured JSON with bounding boxes depending on task

## Architecture

Florence-2 uses a unified sequence-to-sequence architecture with:
- Vision encoder for image understanding
- Language decoder for text generation
- Task-specific prompts for different capabilities
- Support for both text and structured outputs (bboxes, labels)

## Notes

- The model uses `trust_remote_code=True` for loading custom code
- Task prompts like `<OCR>` and `<CAPTION>` are required - the model was trained with these special tokens
- GPU recommended (works on CPU but slower)
- Uses float16 on CUDA, float32 on CPU
- Object detection and OCR with regions return structured JSON with bounding box coordinates
- Model caches to `~/.cache/huggingface` and persists across runs

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
