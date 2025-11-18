# MiniCPM-V-2.6

MiniCPM-V-2.6 is a vision-language model from OpenBMB with 8B parameters (SigLip-400M vision encoder + Qwen2-7B language model).

## Model Information

- **Model**: `openbmb/MiniCPM-V-2_6`
- **Size**: 8B parameters (SigLip-400M + Qwen2-7B)
- **VRAM**: ~16GB
- **License**: Apache 2.0
- **Released**: 2024
- **Project**: [MiniCPM GitHub](https://github.com/OpenBMB/MiniCPM-V)

## Features

### Key Capabilities

- **OCR**: Text extraction
- **Image Captioning**: Image descriptions
- **Visual QA**: Question answering
- **Multi-Image Support**: Compare and analyze multiple images
- **Video Understanding**: Process video frames (with decord)
- **Multilingual**: Supports 30+ languages

## Authentication Required

**Important**: MiniCPM-V-2.6 is a gated model on HuggingFace. You must:
1. Create a HuggingFace account at https://huggingface.co
2. Request access to the model at https://huggingface.co/openbmb/MiniCPM-V-2_6
3. Create an access token at https://huggingface.co/settings/tokens
4. Set the token: `export HUGGING_FACE_HUB_TOKEN=your_token_here`

## Quick Start

### 1. Build the Docker Image

```bash
bash build.sh
```

This creates a ~20GB Docker image with all dependencies including decord for video support.

### 2. Authenticate with HuggingFace

```bash
# Set your HuggingFace token
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

### 3. Run Inference

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
bash predict.sh --image photo.jpg --task vqa --prompt "What is happening in this scene?"
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

**Document OCR:**
```bash
bash predict.sh --image invoice.jpg --task ocr --format json
```

**Image Description:**
```bash
bash predict.sh --image scene.jpg --task caption
```

**Custom Questions:**
```bash
bash predict.sh \
  --image diagram.png \
  --task vqa \
  --prompt "Explain the workflow shown in this diagram"
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

This example uses shared test images from `../test_images/` to avoid file duplication. See `../test_images/README.md`.

## Performance

Tested on NVIDIA A10 GPU with invoice test image (800x600):

- **VRAM Usage**: ~16GB with bfloat16
- **Model Download Size**: ~16GB
- **OCR Accuracy**: 100% accurate on invoice test image with excellent formatting preservation
- **Gated Model**: Requires HuggingFace authentication (HF_TOKEN)

## Technical Details

### Architecture

- **Vision Encoder**: SigLip-400M
- **Language Model**: Qwen2-7B
- **Total Parameters**: 8B
- **Attention**: SDPA (Scaled Dot-Product Attention)
- **Precision**: BFloat16

### Message Format

MiniCPM-V uses a simple message format:

```python
msgs = [
    {
        'role': 'user',
        'content': [image, "Your question here"]
    }
]
```

### Multi-Image Support

Process multiple images together:

```python
msgs = [
    {
        'role': 'user',
        'content': [image1, image2, "Compare these images"]
    }
]
```

## Special Features

- **Multilingual OCR**: Supports 30+ languages
- **Long Context**: Handles detailed images
- **Video Support**: Process video frames with decord
- **Chat Interface**: Natural conversation format
- **SDPA Attention**: Scaled Dot-Product Attention for efficiency

## Requirements

- GPU highly recommended (works on CPU but slower)
- CUDA for optimal performance
- BFloat16 support for best efficiency

## Notes

- Uses custom `chat` method instead of standard `generate`
- SDPA attention provides good balance of speed and memory
- Supports both local images and URLs
- Gated model requires HuggingFace access token
- Tested with 100% accuracy on invoice OCR task

## Citation

```bibtex
@article{minicpm-v-2-6,
  title={MiniCPM-V 2.6: A GPT-4V Level MLLM for Single Image, Multi Image and Video on Your Phone},
  author={OpenBMB Team},
  year={2024}
}
```

## License

Apache 2.0 - See model card at https://huggingface.co/openbmb/MiniCPM-V-2_6
