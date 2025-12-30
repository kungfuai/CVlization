FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install docling and docling-serve
RUN pip install --no-cache-dir \
    docling \
    docling-ibm-models \
    fastapi \
    uvicorn \
    python-multipart

# Pre-download RapidOCR PyTorch models during build
# Initialize DocumentConverter which triggers RapidOCR torch model downloads
RUN python3 -c "from docling.document_converter import DocumentConverter; DocumentConverter()" || true

# Create directories
RUN mkdir -p /app/inputs /app/outputs

# Copy prediction script
COPY predict.py /app/

# Expose port for API (optional, if you want to run as a service later)
EXPOSE 8000

# Default command runs the predict script
CMD ["python", "predict.py"]
