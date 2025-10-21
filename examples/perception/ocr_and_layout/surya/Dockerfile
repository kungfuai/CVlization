FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Surya requires torch>=2.7.0, which will upgrade to 2.8.0
# We need to also upgrade torchvision to match
RUN pip install --no-cache-dir \
    surya-ocr==0.17.0 \
    pillow \
    requests && \
    pip install --no-cache-dir --upgrade torchvision

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
