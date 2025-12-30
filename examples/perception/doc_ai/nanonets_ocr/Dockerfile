FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install transformers and required packages
# Nanonets-OCR2-3B uses qwen2_5_vl architecture (model created with transformers 4.55.4)
RUN pip install --no-cache-dir \
    transformers==4.55.4 \
    accelerate==1.2.1 \
    pillow==11.1.0 \
    pdf2image==1.17.0 \
    qwen-vl-utils==0.0.8

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
