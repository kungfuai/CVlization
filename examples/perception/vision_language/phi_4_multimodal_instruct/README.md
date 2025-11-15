# Phi-4-Multimodal-Instruct

Microsoft's Phi-4-Multimodal-Instruct is a 5.6B parameter vision-language model with state-of-the-art capabilities in OCR, document understanding, visual question answering, and speech recognition.

**Model**: `microsoft/Phi-4-multimodal-instruct`
**Parameters**: 5.6B
**Modalities**: Vision, Text, Audio
**Context**: Up to 128k tokens (hardware dependent)

## Features

- **OCR & Document Understanding**: Excellent text extraction with clean formatting
- **Visual Question Answering**: Answer questions about images
- **Multi-Image Reasoning**: Compare and reason across multiple images
- **Speech Recognition & Translation**: #1 on OpenASR leaderboard
- **128k Context Window**: Process long documents (requires sufficient VRAM)

## Quick Start

### 1. Build the Docker Image

```bash
bash build.sh
```

This builds a Docker image with PyTorch 2.6.0, transformers 4.48.2, and vLLM 0.8.0.

### 2. Run Inference (Standalone)

```bash
cvl run phi-4-multimodal-instruct predict --image path/to/image.jpg --prompt "Describe this image"
```

Or use the script directly:
```bash
bash predict.sh --image test_image.png --prompt "What is in this image?"
```

### 3. Start vLLM Server

```bash
cvl run phi-4-multimodal-instruct serve
```

The server starts on `http://localhost:8000` with OpenAI-compatible API.

## vLLM Server Configuration

### Memory Requirements

The context length directly affects VRAM usage:

| Context Length | VRAM Required | GPU Examples | Setting |
|---------------|---------------|--------------|---------|
| 8k tokens | ~14 GB | RTX 4090, RTX 3090 | `PHI4_MAX_MODEL_LEN=8192` |
| **16k tokens** | ~18 GB | RTX 4090, A10 | `PHI4_MAX_MODEL_LEN=16384` (default) |
| 32k tokens | ~26 GB | A100 40GB | `PHI4_MAX_MODEL_LEN=32768` |
| 128k tokens | ~48 GB | A100 80GB, H100 | `PHI4_MAX_MODEL_LEN=131072` |

**Default**: 16k context (fits on 22GB GPUs like RTX 4090, A10)

### Environment Variables

Configure the vLLM server with these environment variables:

```bash
# Context length (default: 16384)
export PHI4_MAX_MODEL_LEN=16384

# Port (default: 8000)
export PORT=8080

# Model name served via API (default: phi-4-multimodal)
export PHI4_SERVED_NAME=phi-4-multimodal

# Tensor parallel size for multi-GPU (default: 1)
export TENSOR_PARALLEL_SIZE=1

# Additional vLLM arguments
export PHI4_EXTRA_SERVE_ARGS="--gpu-memory-utilization 0.9"

# HuggingFace token (if needed)
export HF_TOKEN=your_token_here
```

### Example: 32k Context on A100

```bash
PHI4_MAX_MODEL_LEN=32768 cvl run phi-4-multimodal-instruct serve
```

### Example: Multi-GPU Setup

```bash
TENSOR_PARALLEL_SIZE=2 cvl run phi-4-multimodal-instruct serve
```

## Using the OpenAI-Compatible API

Once the server is running, use it like OpenAI's API:

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require auth
)

# Encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Chat completion with vision
response = client.chat.completions.create(
    model="phi-4-multimodal",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            }
        ]
    }],
    max_tokens=1024
)

print(response.choices[0].message.content)
```

## Hardware Requirements

**Minimum (Inference)**:
- GPU: 12 GB VRAM (RTX 3090, RTX 4080)
- Context: 8k tokens

**Recommended (Serving)**:
- GPU: 24 GB VRAM (RTX 4090, A10, RTX 6000)
- Context: 16k tokens

**Optimal (Long Context)**:
- GPU: 40-80 GB VRAM (A100, H100)
- Context: 32k-128k tokens

## Troubleshooting

### Out of Memory Error

If you see `CUDA out of memory`, reduce the context length:

```bash
PHI4_MAX_MODEL_LEN=8192 cvl run phi-4-multimodal-instruct serve
```

### Port Already in Use

Change the port:

```bash
PORT=8080 cvl run phi-4-multimodal-instruct serve
```

### Model Download Issues

Set your HuggingFace token:

```bash
export HF_TOKEN=your_token_here
cvl run phi-4-multimodal-instruct serve
```

## Model Information

- **Paper**: [Microsoft Phi-4 Technical Report](https://arxiv.org/abs/2412.08905)
- **HuggingFace**: [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
- **License**: MIT

## Verification

- **Last Verified**: 2025-11-12
- **Hardware**: NVIDIA A10 (23GB VRAM)
- **Status**: ✅ VERIFIED working
- **Performance**: Excellent OCR quality with perfect accuracy and clean formatting
