# Phi-3.5-vision-instruct - Microsoft's Multimodal LLM

Phi-3.5-vision-instruct is a powerful multimodal model from Microsoft with 4.2B parameters, offering strong reasoning capabilities and 128K context length for vision-language tasks.

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

- **Image Captioning**: Detailed image descriptions
- **OCR**: Text extraction from images
- **Visual Question Answering**: Answer questions about images
- **Visual Reasoning**: Complex reasoning about visual content
- **Multi-image Support**: Process multiple images in conversation

### Key Strengths

- Strong reasoning abilities
- High-quality responses
- 128K token context (supports long conversations)
- Fast inference with Flash Attention
- Multi-turn conversations with images

## Quick Start

### 1. Build the Docker Image

```bash
bash build.sh
```

This creates a ~15GB Docker image with Flash Attention support.

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

- **VRAM Usage**: ~8GB
- **Model Size**: 8.3GB on disk
- **Speed**: Fast with Flash Attention 2
- **Context**: Supports up to 128K tokens
- **Quality**: Strong reasoning and detailed responses

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

- The model uses `trust_remote_code=True` for loading
- Flash Attention 2 is recommended for optimal performance
- Use `num_crops=16` for single images (default)
- Use `num_crops=4` for multi-frame/video scenarios
- Model excels at reasoning and detailed analysis
- Works well for document understanding and complex VQA

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
