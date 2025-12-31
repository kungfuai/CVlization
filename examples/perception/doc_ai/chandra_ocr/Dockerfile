FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Chandra OCR and dependencies
RUN pip install --no-cache-dir \
    chandra-ocr==0.1.8 \
    transformers \
    accelerate \
    Pillow \
    numpy

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
