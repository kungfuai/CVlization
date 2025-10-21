FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (GPU version)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install transformers and required packages
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    accelerate \
    pillow \
    pdf2image \
    qwen-vl-utils

# Create directories
RUN mkdir -p /app/inputs /app/outputs /app/examples

# Copy prediction script
COPY predict.py /app/

# Default command runs the predict script
CMD ["python", "predict.py", "--help"]
