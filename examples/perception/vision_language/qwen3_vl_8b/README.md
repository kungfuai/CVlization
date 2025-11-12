# Qwen3-VL-8B-Instruct

Qwen3-VL-8B-Instruct is a vision-language model from Alibaba Cloud with 8B parameters. It is the largest model in the Qwen3-VL family.

## Model Information

- **Model**: `Qwen/Qwen3-VL-8B-Instruct`
- **Size**: 8B parameters
- **VRAM**: ~16GB
- **License**: Apache 2.0
- **Released**: October 15, 2025
- **Paper**: [Qwen3-VL Technical Report](https://qwenlm.github.io/blog/qwen3-vl/)

## Features

### Key Capabilities

- **OCR**: Text recognition and extraction
- **Visual Reasoning**: Understanding and logical analysis
- **Image Captioning**: Image descriptions
- **Video Understanding**: Process video frames and sequences
- **Multi-image Support**: Analyze multiple images together
- **Extended Context**: Long context understanding

## Quick Start

### 1. Build the Docker Image

```bash
bash build.sh
```

This creates a Docker image with the latest transformers from source.

**Note**: Build may take longer as it installs transformers from GitHub for Qwen3-VL support.

### 2. Run Inference

#### Image Captioning

```bash
bash predict.sh --image test_images/sample.jpg --task caption
```

#### OCR (Text Extraction)

```bash
bash predict.sh --image document.jpg --task ocr
```

#### Visual Question Answering

```bash
bash predict.sh --image photo.jpg --task vqa --prompt "What is the main subject of this image?"
```

### 3. Run Tests

```bash
bash test.sh
```

## Usage

### Basic Usage

```bash
bash predict.sh --image <path_or_url> --task <task_name>
```

### All Options

```bash
bash predict.sh \
  --image <path_or_url> \
  --task <caption|ocr|vqa> \
  --prompt <custom_prompt> \
  --output <output_path> \
  --format <txt|json> \
  --device <cuda|mps|cpu>
```

### Task-Specific Examples

**Detailed Captioning:**
```bash
bash predict.sh --image landscape.jpg --task caption
```

**Document OCR:**
```bash
bash predict.sh --image invoice.pdf --task ocr --format json
```

**Visual Reasoning:**
```bash
bash predict.sh \
  --image diagram.png \
  --task vqa \
  --prompt "Explain the relationships between the components in this diagram"
```

**Image from URL:**
```bash
bash predict.sh \
  --image https://example.com/photo.jpg \
  --task caption
```

## Output Formats

### Text Output (default)

Simple text results:

```bash
bash predict.sh --image photo.jpg --task caption --output result.txt
```

### JSON Output

Structured output with metadata:

```bash
bash predict.sh --image doc.jpg --task ocr --output result.json --format json
```

## Test Images

This example uses shared test images from `../test_images/` to avoid file duplication. See `../test_images/README.md`.

## Performance

Tested on NVIDIA A10 GPU with invoice test image (800x600):

- **VRAM Usage**: ~16GB
- **Model Download Size**: ~16GB
- **Notes**: Largest model in Qwen3-VL family, provides maximum capability at the cost of higher resource usage

## Technical Details

### Message Format

Qwen3-VL uses a structured message format with content arrays:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_or_url},
            {"type": "text", "text": "Your question here"},
        ],
    }
]
```

### Multi-Image Support

Process multiple images in a single query:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image1},
            {"type": "image", "image": image2},
            {"type": "text", "text": "Compare these images"},
        ],
    }
]
```

### Video Understanding

Qwen3-VL can process video frames for temporal understanding.

## Requirements

- Requires latest transformers (installed from GitHub source)
- GPU recommended (works on CPU but much slower)
- Supports both local images and URLs
- Compatible with CVlization dual-mode execution

## Notes

- Uses latest transformers from source for Qwen3-VL support
- Largest model in Qwen3-VL family (2B, 4B, 8B variants available)
- Supports both images and videos
- Works with URLs and local files
- Provides maximum capability at higher resource cost
- Supports multi-turn conversation

## Citation

```bibtex
@article{qwen3vl2025,
  title={Qwen3-VL: Scaling Up Vision Language Models},
  author={Qwen Team},
  journal={arXiv preprint},
  year={2025}
}
```

## License

Apache 2.0 - See model card at https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
