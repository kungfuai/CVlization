FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

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

# Install transformers and required packages (torch already included in base image)
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    pillow

# Create directories
RUN mkdir -p /app/inputs /app/outputs

# Copy prediction scripts
COPY predict.py batch_predict.py /app/

# Default command runs the predict script
CMD ["python", "predict.py"]
