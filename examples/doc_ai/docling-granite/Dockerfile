FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version for smaller image, use GPU version if needed)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install transformers and required packages
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    pillow \
    qwen-vl-utils

# Create directories
RUN mkdir -p /app/inputs /app/outputs

# Copy prediction script
COPY predict.py /app/

# Default command runs the predict script
CMD ["python", "predict.py"]
