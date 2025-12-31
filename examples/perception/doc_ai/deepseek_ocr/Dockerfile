# PyTorch 2.9.1 with CUDA 12.8 for Blackwell GPU support
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies for DeepSeek-OCR
# Based on requirements.txt from https://github.com/deepseek-ai/DeepSeek-OCR
RUN pip install --no-cache-dir \
    transformers==4.46.3 \
    tokenizers==0.20.3 \
    einops \
    easydict \
    addict \
    Pillow \
    numpy \
    PyMuPDF \
    img2pdf \
    matplotlib \
    accelerate

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
