FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# surya-ocr 0.17.0 requires pillow<11.0.0
RUN pip install --no-cache-dir \
    surya-ocr==0.17.0 \
    pillow==10.4.0 \
    requests==2.32.3

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
