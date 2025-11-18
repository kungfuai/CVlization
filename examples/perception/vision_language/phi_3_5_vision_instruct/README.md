# Phi-3.5-vision-instruct

Phi-3.5-vision-instruct is a multimodal model from Microsoft with 4.2B parameters and 128K context length for vision-language tasks.

## Model Information

- **Model**: `microsoft/Phi-3.5-vision-instruct`
- **Size**: 4.2B parameters
- **VRAM**: ~8GB
- **Context**: 128K tokens
- **License**: MIT
- **Released**: August 2024
- **Paper**: [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)

## Features

### Capabilities

- **Image Captioning**: Image descriptions
- **OCR**: Text extraction from images
- **Visual Question Answering**: Answer questions about images
- **Visual Reasoning**: Reasoning about visual content
- **Multi-image Support**: Process multiple images in conversation

### Key Features

- 128K token context (supports long conversations)
- Flash Attention 2 support for faster inference
- Multi-turn conversations with images

## Quick Start

### 1. Build the Docker Image

```bash
bash build.sh
```

This creates a Docker image with Flash Attention 2 support.

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
bash predict.sh --image photo.jpg --task vqa --prompt "What is happening in this image?"
```

#### Custom Questions

```bash
bash predict.sh --image chart.png --task vqa --prompt "What trends do you see in this data?"
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
  --task <caption|ocr|vqa> \
  --prompt <custom_prompt> \
  --output <output_path> \
  --format <txt|json> \
  --device <cuda|mps|cpu>
```

### Task-Specific Examples

**Image Captioning:**
```bash
bash predict.sh --image photo.jpg --task caption
```

**Document OCR:**
```bash
bash predict.sh --image receipt.jpg --task ocr --format json
```

**Custom VQA:**
```bash
bash predict.sh \
  --image diagram.png \
  --task vqa \
  --prompt "Explain the components in this system diagram"
```

## Output Formats

### Text Output (default)

Simple text results saved to file:

```bash
bash predict.sh --image photo.jpg --task caption --output result.txt
```

### JSON Output

Structured output with metadata:

```bash
bash predict.sh --image doc.jpg --task ocr --output result.json --format json
```

## Test Images

This example uses shared test images from `../test_images/` to avoid duplicating image files. See `../test_images/README.md` for details.

## Performance

Tested on NVIDIA A10 GPU with invoice test image (800x600):

- **VRAM Usage**: ~8GB
- **Model Download Size**: ~8GB
- **Context Length**: Supports up to 128K tokens
- **Flash Attention**: Uses Flash Attention 2 for optimized inference

## Technical Details

### Chat Format

Phi-3.5-vision uses a chat format with image placeholders:

```python
messages = [
    {"role": "user", "content": "<|image_1|>\nWhat is in this image?"},
]
```

### Attention Mechanisms

- **Flash Attention 2** (default): Fastest, requires flash-attn package
- **Eager Attention** (fallback): Works without flash-attn but slower

The script automatically detects Flash Attention availability and uses the appropriate implementation.

### Multi-Image Support

For processing multiple images:

```python
messages = [
    {"role": "user", "content": "<|image_1|>\n<|image_2|>\nCompare these images."},
]
```

Image indices start from 1.

## Notes

- The model uses `trust_remote_code=True` for loading custom code
- Flash Attention 2 is recommended for optimal performance
- Use `num_crops=16` for single images (default)
- Use `num_crops=4` for multi-frame/video scenarios
- Supports document understanding and visual question answering
- Model caches to `~/.cache/huggingface` and persists across runs

## Citation

```bibtex
@article{abdin2024phi,
  title={Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone},
  author={Abdin, Marah and others},
  journal={arXiv preprint arXiv:2404.14219},
  year={2024}
}
```

## License

MIT License - See model card at https://huggingface.co/microsoft/Phi-3.5-vision-instruct
